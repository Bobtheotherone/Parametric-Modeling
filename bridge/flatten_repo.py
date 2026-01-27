#!/usr/bin/env python3
"""
Flatten Repository to Text File

This script consolidates all relevant source files in a repository into a single
text file for easy sharing, code review, or feeding into LLMs. It automatically
excludes binary files, caches, virtual environments, and other non-essential files.

Usage:
    Place this script at the root of your repository and run:
    $ python flatten_repo.py

    Optional arguments:
    $ python flatten_repo.py --output custom_name.txt --max-size 2000000
"""

import os
import argparse
from pathlib import Path
from typing import Set

# Directories to always exclude
EXCLUDED_DIRS: Set[str] = {
    # Version control
    '.git',
    '.svn',
    '.hg',
    '.bzr',
    
    # Python
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.tox',
    '.nox',
    '.eggs',
    '*.egg-info',
    'venv',
    '.venv',
    'env',
    '.env',
    'virtualenv',
    '.virtualenv',
    
    # JavaScript/Node
    'node_modules',
    '.npm',
    '.yarn',
    '.pnpm-store',
    'bower_components',
    
    # Build outputs
    'build',
    'dist',
    'out',
    'target',
    'bin',
    'obj',
    '_build',
    
    # IDE and editor
    '.idea',
    '.vscode',
    '.vs',
    '.eclipse',
    '.settings',
    '*.xcworkspace',
    '*.xcodeproj',
    
    # OS files
    '.DS_Store',
    'Thumbs.db',
    
    # Dependencies and packages
    'vendor',
    'packages',
    '.bundle',
    
    # Coverage and testing
    'coverage',
    '.coverage',
    'htmlcov',
    '.nyc_output',
    
    # Logs and temp
    'logs',
    'tmp',
    'temp',
    '.tmp',
    '.temp',
    '.cache',
    
    # Documentation builds
    '_site',
    '.docusaurus',
    '.next',
    '.nuxt',
    '.output',
    
    # Misc
    '.terraform',
    '.serverless',
    '.aws-sam',
}

# File extensions to exclude (binary and non-essential)
EXCLUDED_EXTENSIONS: Set[str] = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.tiff', '.tif', '.psd', '.ai', '.eps',
    
    # Audio/Video
    '.mp3', '.mp4', '.wav', '.flac', '.ogg', '.avi', '.mov', '.mkv',
    '.webm', '.m4a', '.m4v',
    
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
    
    # Compiled/Binary
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe', '.bin',
    '.o', '.a', '.lib', '.class', '.jar', '.war', '.ear',
    
    # Documents (often binary)
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp',
    
    # Fonts
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    
    # Database
    '.db', '.sqlite', '.sqlite3', '.mdb',
    
    # Lock files (often very long and not useful for review)
    '.lock',
    
    # Misc binary
    '.pkl', '.pickle', '.npy', '.npz', '.h5', '.hdf5',
    '.parquet', '.feather', '.arrow',
    '.wasm', '.map',
}

# Specific filenames to exclude
EXCLUDED_FILES: Set[str] = {
    # Lock files
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml',
    'Pipfile.lock',
    'poetry.lock',
    'composer.lock',
    'Gemfile.lock',
    'Cargo.lock',
    
    # Generated files
    '.gitattributes',
    '.editorconfig',
    '.prettierrc',
    '.eslintcache',
    
    # This script itself
    'flatten_repo.py',
    
    # OS files
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini',
}

# Extensions that are definitely text (for verification)
TEXT_EXTENSIONS: Set[str] = {
    # Programming languages
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc',
    '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
    '.kts', '.scala', '.clj', '.cljs', '.erl', '.ex', '.exs', '.elm',
    '.hs', '.lua', '.pl', '.pm', '.r', '.R', '.jl', '.nim', '.zig',
    '.v', '.vhdl', '.verilog', '.sol', '.vy', '.move', '.cairo',
    
    # Web
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
    
    # Data/Config
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.xml', '.csv', '.tsv',
    
    # Documentation
    '.md', '.markdown', '.rst', '.txt', '.adoc', '.asciidoc', '.org',
    
    # Shell/Scripts
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    
    # Build/Config files
    '.dockerfile', '.containerfile', '.tf', '.hcl',
    '.cmake', '.make', '.mk', '.gradle', '.sbt',
    
    # Misc text
    '.sql', '.graphql', '.gql', '.proto', '.thrift', '.avsc',
    '.env.example', '.env.sample', '.env.template',
}

# Files without extensions that are typically text
TEXT_FILENAMES: Set[str] = {
    'Dockerfile',
    'Containerfile',
    'Makefile',
    'CMakeLists.txt',
    'Rakefile',
    'Gemfile',
    'Pipfile',
    'Procfile',
    'Vagrantfile',
    'Brewfile',
    'Justfile',
    '.gitignore',
    '.dockerignore',
    '.npmignore',
    '.prettierignore',
    '.eslintignore',
    'LICENSE',
    'LICENCE',
    'README',
    'CHANGELOG',
    'CONTRIBUTING',
    'AUTHORS',
    'MAINTAINERS',
    'CODEOWNERS',
    'requirements.txt',
    'setup.py',
    'setup.cfg',
    'pyproject.toml',
}


def should_exclude_dir(dir_name: str) -> bool:
    """Check if a directory should be excluded."""
    # Check exact matches
    if dir_name in EXCLUDED_DIRS:
        return True
    # Check if starts with dot (hidden directories)
    if dir_name.startswith('.') and dir_name not in {'.github', '.circleci', '.gitlab'}:
        return True
    # Check patterns (e.g., *.egg-info)
    if dir_name.endswith('.egg-info'):
        return True
    return False


def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded."""
    file_name = file_path.name
    extension = file_path.suffix.lower()
    
    # Check excluded filenames
    if file_name in EXCLUDED_FILES:
        return True
    
    # Check excluded extensions
    if extension in EXCLUDED_EXTENSIONS:
        return True
    
    # Check if hidden file (but allow some important ones)
    if file_name.startswith('.') and file_name not in {'.gitignore', '.dockerignore', '.env.example'}:
        return True
    
    return False


def is_likely_text_file(file_path: Path, max_check_bytes: int = 8192) -> bool:
    """
    Check if a file is likely a text file by examining its content.
    Returns True if the file appears to be text, False if binary.
    """
    file_name = file_path.name
    extension = file_path.suffix.lower()
    
    # Known text extensions
    if extension in TEXT_EXTENSIONS:
        return True
    
    # Known text filenames
    if file_name in TEXT_FILENAMES:
        return True
    
    # For unknown extensions, check file content
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(max_check_bytes)
        
        if not chunk:  # Empty file
            return True
        
        # Check for null bytes (strong indicator of binary)
        if b'\x00' in chunk:
            return False
        
        # Try to decode as UTF-8
        try:
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    chunk.decode(encoding)
                    # Additional check: high ratio of printable characters
                    text = chunk.decode(encoding)
                    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
                    if printable_ratio > 0.85:
                        return True
                except:
                    continue
            return False
    except (IOError, OSError):
        return False


def get_file_separator(file_path: str) -> str:
    """Generate a visual separator for a file in the output."""
    separator = "=" * 80
    return f"\n{separator}\n FILE: {file_path}\n{separator}\n\n"


def flatten_repo(
    root_dir: Path,
    output_file: Path,
    max_file_size: int = 1_000_000,  # 1MB default
    max_total_size: int = 50_000_000,  # 50MB default
) -> dict:
    """
    Flatten all relevant files in a repository into a single text file.
    
    Args:
        root_dir: Root directory of the repository
        output_file: Path to the output text file
        max_file_size: Maximum size of individual files to include (bytes)
        max_total_size: Maximum total size of output file (bytes)
    
    Returns:
        Statistics about the flattening process
    """
    stats = {
        'files_included': 0,
        'files_excluded': 0,
        'dirs_excluded': 0,
        'total_bytes': 0,
        'excluded_reasons': {},
    }
    
    included_files = []
    
    # Walk the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to skip excluded directories
        excluded_dirs = [d for d in dirnames if should_exclude_dir(d)]
        for d in excluded_dirs:
            dirnames.remove(d)
            stats['dirs_excluded'] += 1
        
        # Sort remaining directories for consistent output
        dirnames.sort()
        
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            relative_path = file_path.relative_to(root_dir)
            
            # Skip the output file itself
            if file_path.resolve() == output_file.resolve():
                continue
            
            # Check if file should be excluded
            if should_exclude_file(file_path):
                stats['files_excluded'] += 1
                reason = f"excluded pattern: {file_path.suffix or file_path.name}"
                stats['excluded_reasons'][reason] = stats['excluded_reasons'].get(reason, 0) + 1
                continue
            
            # Check file size
            try:
                file_size = file_path.stat().st_size
            except OSError:
                stats['files_excluded'] += 1
                continue
            
            if file_size > max_file_size:
                stats['files_excluded'] += 1
                stats['excluded_reasons']['too large'] = stats['excluded_reasons'].get('too large', 0) + 1
                continue
            
            if file_size == 0:
                stats['files_excluded'] += 1
                stats['excluded_reasons']['empty'] = stats['excluded_reasons'].get('empty', 0) + 1
                continue
            
            # Check if file is text
            if not is_likely_text_file(file_path):
                stats['files_excluded'] += 1
                stats['excluded_reasons']['binary'] = stats['excluded_reasons'].get('binary', 0) + 1
                continue
            
            included_files.append((relative_path, file_path, file_size))
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8', errors='replace') as out:
        # Write header
        out.write("=" * 80 + "\n")
        out.write(" FLATTENED REPOSITORY\n")
        out.write(f" Source: {root_dir.resolve()}\n")
        out.write(f" Files included: {len(included_files)}\n")
        out.write("=" * 80 + "\n\n")
        
        # Write table of contents
        out.write("TABLE OF CONTENTS\n")
        out.write("-" * 40 + "\n")
        for rel_path, _, _ in included_files:
            out.write(f"  - {rel_path}\n")
        out.write("\n")
        
        # Write file contents
        for rel_path, file_path, file_size in included_files:
            # Check total size limit
            if stats['total_bytes'] + file_size > max_total_size:
                out.write(f"\n[OUTPUT TRUNCATED - reached {max_total_size:,} byte limit]\n")
                break
            
            out.write(get_file_separator(str(rel_path)))
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                out.write(content)
                if not content.endswith('\n'):
                    out.write('\n')
                stats['files_included'] += 1
                stats['total_bytes'] += file_size
            except Exception as e:
                out.write(f"[Error reading file: {e}]\n")
                stats['files_excluded'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Flatten a repository into a single text file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flatten_repo.py
  python flatten_repo.py --output repo_contents.txt
  python flatten_repo.py --max-size 500000 --max-total 10000000
        """
    )
    parser.add_argument(
        '--output', '-o',
        default='repo_flattened.txt',
        help='Output file name (default: repo_flattened.txt)'
    )
    parser.add_argument(
        '--max-size', '-s',
        type=int,
        default=1_000_000,
        help='Maximum size of individual files in bytes (default: 1MB)'
    )
    parser.add_argument(
        '--max-total', '-t',
        type=int,
        default=50_000_000,
        help='Maximum total output size in bytes (default: 50MB)'
    )
    parser.add_argument(
        '--root', '-r',
        default='.',
        help='Root directory to flatten (default: current directory)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed statistics'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    output_file = root_dir / args.output
    
    print(f"Flattening repository: {root_dir}")
    print(f"Output file: {output_file}")
    print()
    
    stats = flatten_repo(
        root_dir=root_dir,
        output_file=output_file,
        max_file_size=args.max_size,
        max_total_size=args.max_total,
    )
    
    print(f"✓ Done!")
    print(f"  Files included: {stats['files_included']}")
    print(f"  Files excluded: {stats['files_excluded']}")
    print(f"  Directories skipped: {stats['dirs_excluded']}")
    print(f"  Total size: {stats['total_bytes']:,} bytes")
    
    if args.verbose and stats['excluded_reasons']:
        print(f"\nExclusion breakdown:")
        for reason, count in sorted(stats['excluded_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    print(f"\nOutput written to: {output_file}")


if __name__ == '__main__':
    main()