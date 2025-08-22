use crate::ffi::{FileFormat, check_fmt};
use base::libc::{O_RDONLY, O_TRUNC, O_WRONLY};
use base::{Chunker, LoggedResult, Utf8CStr, WriteExt, error, log_err};
use bytemuck::bytes_of_mut;
use bzip2::{Compression as BzCompression, read::BzDecoder, write::BzEncoder};
use flate2::{Compression as GzCompression, write::GzEncoder, read::MultiGzDecoder};
use lz4::{
    BlockMode, BlockSize, ContentChecksum, Encoder as LZ4FrameEncoder,
    EncoderBuilder as LZ4FrameEncoderBuilder, block::CompressionMode, liblz4::BlockChecksum,
};
use lzma_rust2::{CheckType, LZMAOptions, LZMAReader, LZMAWriter, XZOptions, XZReader, XZWriter};
use std::fs::File;
use std::io::{BufWriter, Read, Write, stdin, stdout};
use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::ops::DerefMut;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, RawFd};
use zopfli::{BlockType, GzipEncoder as ZopFliEncoder, Options as ZopfliOptions};

pub trait WriteFinish<W: Write>: Write {
    fn finish(self: Box<Self>) -> std::io::Result<W>;
}

// Boilerplate for existing types

macro_rules! finish_impl {
    ($($t:ty),*) => {$(
        impl<W: Write> WriteFinish<W> for $t {
            fn finish(self: Box<Self>) -> std::io::Result<W> {
                Self::finish(*self)
            }
        }
    )*}
}

finish_impl!(
    GzEncoder<W>,
    BzEncoder<W>,
    XZWriter<W>,
    LZMAWriter<W>
);

impl<W: Write> WriteFinish<W> for BufWriter<ZopFliEncoder<W>> {
    fn finish(self: Box<Self>) -> std::io::Result<W> {
        let inner = self.into_inner()?;
        ZopFliEncoder::finish(inner)
    }
}

impl<W: Write> WriteFinish<W> for LZ4FrameEncoder<W> {
    fn finish(self: Box<Self>) -> std::io::Result<W> {
        let (w, r) = Self::finish(*self);
        r?;
        Ok(w)
    }
}

// LZ4BlockArchive format
//
// len:  |   4   |          4            |           n           | ... |           4             |
// data: | magic | compressed block size | compressed block data | ... | total uncompressed size |

// LZ4BlockEncoder

const LZ4_BLOCK_SIZE: usize = 0x800000;
const LZ4HC_CLEVEL_MAX: i32 = 12;
const LZ4_MAGIC: u32 = 0x184c2102;

struct LZ4LegacyEncoder<W: Write> {
    write: W,
    chunker: Chunker,
    out_buf: Box<[u8]>,
    total: u32,
    is_lg: bool,
}

impl<W: Write> LZ4LegacyEncoder<W> {
    fn new(write: W, is_lg: bool) -> Self {
        let out_sz = lz4::block::compress_bound(LZ4_BLOCK_SIZE).unwrap_or(LZ4_BLOCK_SIZE);
        Self {
            write,
            chunker: Chunker::new(LZ4_BLOCK_SIZE),
            // SAFETY: all bytes will be initialized before it is used
            out_buf: unsafe { Box::new_uninit_slice(out_sz).assume_init() },
            total: 0,
            is_lg,
        }
    }

    fn encode_block(write: &mut W, out_buf: &mut [u8], chunk: &[u8]) -> std::io::Result<()> {
        let compressed_size = lz4::block::compress_to_buffer(
            chunk,
            Some(CompressionMode::HIGHCOMPRESSION(LZ4HC_CLEVEL_MAX)),
            false,
            out_buf,
        )?;
        let block_size = compressed_size as u32;
        write.write_pod(&block_size)?;
        write.write_all(&out_buf[..compressed_size])
    }
}

impl<W: Write> Write for LZ4LegacyEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, mut buf: &[u8]) -> std::io::Result<()> {
        if self.total == 0 {
            // Write header
            self.write.write_pod(&LZ4_MAGIC)?;
        }

        self.total += buf.len() as u32;
        while !buf.is_empty() {
            let (b, chunk) = self.chunker.add_data(buf);
            buf = b;
            if let Some(chunk) = chunk {
                Self::encode_block(&mut self.write, &mut self.out_buf, chunk)?;
            }
        }
        Ok(())
    }
}

impl<W: Write> WriteFinish<W> for LZ4LegacyEncoder<W> {
    fn finish(mut self: Box<Self>) -> std::io::Result<W> {
        let chunk = self.chunker.get_available();
        if !chunk.is_empty() {
            Self::encode_block(&mut self.write, &mut self.out_buf, chunk)?;
        }
        if self.is_lg {
            self.write.write_pod(&self.total)?;
        }
        Ok(self.write)
    }
}

// LZ4BlockDecoder

struct LZ4LegacyDecoder<R: Read> {
    read: R,
    remaining: Vec<u8>,
    offset: usize,
}

impl<R: Read> LZ4LegacyDecoder<R> {
    fn new(read: R) -> Self {
         Self {
             read,
             // SAFETY: all bytes will be initialized before it is used
             remaining: vec![],
             offset: 0,
        }
    }
}

impl<R: Read> Read for LZ4LegacyDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.offset == self.remaining.len() {
            let mut block_size: u32 = 0;
            if let Err(e) = self.read.read_exact(bytes_of_mut(&mut block_size)) {
                return if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    Ok(0)
                } else {
                    Err(e)
                }
            }
            if block_size == LZ4_MAGIC {
                self.read.read_exact(bytes_of_mut(&mut block_size))?;
            }
            unsafe { self.remaining.set_len(0) };
            self.remaining.reserve(LZ4_BLOCK_SIZE);
            unsafe { self.remaining.set_len(LZ4_BLOCK_SIZE) };
            let mut test_byte = [0u8; 1];
            if let Ok(0) = self.read.read(&mut test_byte) {
                return Ok(0);
            }
            let mut src = Vec::with_capacity(block_size as usize);
            unsafe { src.set_len(block_size as usize) };
            src[0] = test_byte[0];
            self.read.read_exact(&mut src[1..])?;
            let sz = lz4::block::decompress_to_buffer(&src, Some(LZ4_BLOCK_SIZE as i32), &mut self.remaining)?;
            self.remaining.truncate(sz);
            self.offset = 0;
        }
        let to_copy = (self.remaining.len() - self.offset).min(buf.len());
        buf[..to_copy].copy_from_slice(&self.remaining[self.offset..self.offset + to_copy]);
        self.offset += to_copy;
        Ok(to_copy)
    }
}

// Top-level APIs

pub fn get_encoder<'a, W: Write + 'a>(format: FileFormat, w: W) -> Box<dyn WriteFinish<W> + 'a> {
    match format {
        FileFormat::XZ => {
            let mut opt = XZOptions::with_preset(9);
            opt.set_check_sum_type(CheckType::Crc32);
            Box::new(XZWriter::new(w, opt).unwrap())
        }
        FileFormat::LZMA => {
            Box::new(LZMAWriter::new_use_header(w, &LZMAOptions::with_preset(9), None).unwrap())
        }
        FileFormat::BZIP2 => Box::new(BzEncoder::new(w, BzCompression::best())),
        FileFormat::LZ4 => {
            let encoder = LZ4FrameEncoderBuilder::new()
                .block_size(BlockSize::Max4MB)
                .block_mode(BlockMode::Independent)
                .checksum(ContentChecksum::ChecksumEnabled)
                .block_checksum(BlockChecksum::BlockChecksumEnabled)
                .level(9)
                .auto_flush(true)
                .build(w)
                .unwrap();
            Box::new(encoder)
        }
        FileFormat::LZ4_LEGACY => Box::new(LZ4LegacyEncoder::new(w, false)),
        FileFormat::LZ4_LG => Box::new(LZ4LegacyEncoder::new(w, true)),
        FileFormat::ZOPFLI => {
            // These options are already better than gzip -9
            let opt = ZopfliOptions {
                iteration_count: NonZeroU64::new(1).unwrap(),
                maximum_block_splits: 1,
                ..Default::default()
            };
            Box::new(ZopFliEncoder::new_buffered(opt, BlockType::Dynamic, w).unwrap())
        }
        FileFormat::GZIP => Box::new(GzEncoder::new(w, GzCompression::best())),
        _ => unreachable!(),
    }
}

pub fn get_decoder<'a, R: Read + 'a>(format: FileFormat, r: R) -> Box<dyn Read + 'a> {
    match format {
        FileFormat::XZ => Box::new(XZReader::new(r, true)),
        FileFormat::LZMA => Box::new(LZMAReader::new_mem_limit(r, u32::MAX, None).unwrap()),
        FileFormat::BZIP2 => Box::new(BzDecoder::new(r)),
        FileFormat::LZ4 => Box::new(lz4::Decoder::new(r).unwrap()),
        FileFormat::LZ4_LG | FileFormat::LZ4_LEGACY => Box::new(LZ4LegacyDecoder::new(r)),
        FileFormat::ZOPFLI | FileFormat::GZIP => Box::new(MultiGzDecoder::new(r)),
        _ => unreachable!(),
    }
}

// C++ FFI

pub fn compress_fd(format: FileFormat, in_fd: RawFd, out_fd: RawFd) {
    let mut in_file = unsafe { ManuallyDrop::new(File::from_raw_fd(in_fd)) };
    let mut out_file = unsafe { ManuallyDrop::new(File::from_raw_fd(out_fd)) };

    let mut encoder = get_encoder(format, out_file.deref_mut());
    let _: LoggedResult<()> = try {
        std::io::copy(in_file.deref_mut(), encoder.as_mut())?;
        encoder.finish()?;
    };
}

pub fn decompress_bytes_fd(format: FileFormat, in_bytes: &[u8], in_fd: RawFd, out_fd: RawFd) {
    let mut in_file = unsafe { ManuallyDrop::new(File::from_raw_fd(in_fd)) };
    let mut out_file = unsafe { ManuallyDrop::new(File::from_raw_fd(out_fd)) };

    struct ConcatReader<'a> (&'a [u8], &'a File);

    impl Read for ConcatReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if !self.0.is_empty() {
                let len = self.0.len().min(buf.len());
                buf[..len].copy_from_slice(&self.0[..len]);
                self.0 = &self.0[len..];
                Ok(len)
            } else {
                self.1.read(buf)
            }
        }
    }

    let mut reader = ConcatReader(in_bytes, in_file.deref_mut());
    let mut decoder = get_decoder(format, &mut reader);
    let _: LoggedResult<()> = try {
        std::io::copy(decoder.as_mut(), out_file.deref_mut())?;
    };
}

pub fn compress_bytes(format: FileFormat, in_bytes: &[u8], out_fd: RawFd) {
    let mut out_file = unsafe { ManuallyDrop::new(File::from_raw_fd(out_fd)) };

    let mut encoder = get_encoder(format, out_file.deref_mut());
    let _: LoggedResult<()> = try {
        encoder.write_all(in_bytes)?;
        encoder.finish()?;
    };
}

pub fn decompress_bytes(format: FileFormat, in_bytes: &[u8], out_fd: RawFd) {
    let mut out_file = unsafe { ManuallyDrop::new(File::from_raw_fd(out_fd)) };

    let mut decoder = get_decoder(format, in_bytes);
    let _: LoggedResult<()> = try {
        std::io::copy(decoder.as_mut(), out_file.deref_mut())?;
    };
}

pub(crate) fn decompress(infile: &mut String, outfile: Option<&mut String>) -> LoggedResult<()> {
    let in_std = infile == "-";
    let mut rm_in = false;

    let mut buf = [0u8; 4096];
    let raw_in = if in_std {
        super let mut stdin = stdin();
        let _ = stdin.read(&mut buf)?;
        stdin.as_fd()
    } else {
        super let mut infile = Utf8CStr::from_string(infile).open(O_RDONLY)?;
        let _ = infile.read(&mut buf)?;
        infile.as_fd()
    };

    let format = check_fmt(&buf);

    eprintln!("Detected format: {format}");

    if !format.is_compressed() {
        return log_err!("Input file is not a supported type!");
    }

    let raw_out = if let Some(outfile) = outfile {
        if outfile == "-" {
            super let stdout = stdout();
            stdout.as_fd()
        } else {
            super let outfile = Utf8CStr::from_string(outfile).create(O_WRONLY | O_TRUNC, 0o644)?;
            outfile.as_fd()
        }
    } else if in_std {
        super let stdout = stdout();
        stdout.as_fd()
    } else {
        // strip the extension
        rm_in = true;
        let mut outfile = if let Some((outfile, ext)) = infile.rsplit_once('.') {
            if ext != &format.ext()[1..] {
                return log_err!("Input file is not a supported type!");
            }
            outfile.to_owned()
        } else {
            infile.clone()
        };
        eprintln!("Decompressing to [{outfile}]");

        super let outfile = Utf8CStr::from_string(&mut outfile).create(O_WRONLY | O_TRUNC, 0o644)?;
        outfile.as_fd()
    };

    decompress_bytes_fd(format, &buf, raw_in.as_raw_fd(), raw_out.as_raw_fd());

    if rm_in {
        Utf8CStr::from_string(infile).remove()?;
    }

    Ok(())
}

pub(crate) fn compress(
    method: FileFormat,
    infile: &mut String,
    outfile: Option<&mut String>,
) -> LoggedResult<()> {
    if method == FileFormat::UNKNOWN {
        error!("Unsupported compression format");
    }

    let in_std = infile == "-";
    let mut rm_in = false;

    let raw_in = if in_std {
        super let stdin = stdin();
        stdin.as_fd()
    } else {
        super let infile = Utf8CStr::from_string(infile).open(O_RDONLY)?;
        infile.as_fd()
    };

    let raw_out = if let Some(outfile) = outfile {
        if outfile == "-" {
            super let stdout = stdout();
            stdout.as_fd()
        } else {
            super let outfile = Utf8CStr::from_string(outfile).create(O_WRONLY | O_TRUNC, 0o644)?;
            outfile.as_fd()
        }
    } else if in_std {
        super let stdout = stdout();
        stdout.as_fd()
    } else {
        let mut outfile = format!("{infile}{}", method.ext());
        eprintln!("Compressing to [{outfile}]");
        rm_in = true;
        super let outfile = Utf8CStr::from_string(&mut outfile).create(O_WRONLY | O_TRUNC, 0o644)?;
        outfile.as_fd()
    };

    compress_fd(method, raw_in.as_raw_fd(), raw_out.as_raw_fd());

    if rm_in {
        Utf8CStr::from_string(infile).remove()?;
    }
    Ok(())
}
