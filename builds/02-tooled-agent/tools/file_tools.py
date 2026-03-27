"""
File system tools — read, write, list.

These let the agent interact with the local file system.
Compare to CrewAI's FileReadTool / DirectoryReadTool in their tools package.

Safety note: In production, you'd want sandboxing (restrict paths,
prevent writing to system dirs). We keep it simple for learning.
"""

import os
from .base import tool


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file. Returns the file text. Use this to examine files on disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Truncate very large files to avoid blowing context window
    if len(content) > 10000:
        return content[:10000] + f"\n\n... (truncated, {len(content)} total characters)"
    return content


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Returns confirmation."""
    # Create parent directories if needed
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to {file_path}"


@tool
def list_directory(path: str) -> str:
    """List files and directories at the given path. Returns a formatted listing with file sizes."""
    entries = []
    for entry in sorted(os.listdir(path)):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            entries.append(f"  [DIR]  {entry}/")
        else:
            size = os.path.getsize(full_path)
            entries.append(f"  [FILE] {entry} ({size:,} bytes)")
    return f"Contents of {path}:\n" + "\n".join(entries) if entries else f"{path} is empty"
