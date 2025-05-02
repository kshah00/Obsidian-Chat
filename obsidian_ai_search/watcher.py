import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class MarkdownEventHandler(FileSystemEventHandler):
    """Handles filesystem events for markdown files."""
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).resolve()
        logging.info(f"Initializing watcher for: {self.vault_path}")

    def _is_markdown_file(self, event: FileSystemEvent) -> bool:
        """Check if the event relates to a markdown file within the vault."""
        if event.is_directory:
            return False
        src_path = Path(event.src_path)
        # Check if it's a .md file and within the vault path
        return src_path.suffix.lower() == '.md' and self.vault_path in src_path.parents

    def on_created(self, event: FileSystemEvent):
        if self._is_markdown_file(event):
            logging.info(f"Created: {event.src_path}")
            # TODO: Trigger add file to index

    def on_modified(self, event: FileSystemEvent):
        if self._is_markdown_file(event):
            logging.info(f"Modified: {event.src_path}")
            # TODO: Trigger update file in index

    def on_deleted(self, event: FileSystemEvent):
        if self._is_markdown_file(event):
            logging.info(f"Deleted: {event.src_path}")
            # TODO: Trigger remove file from index

    def on_moved(self, event: FileSystemEvent):
        # Handling moves as delete + create
        if event.is_directory:
            # If a directory is moved, handle all md files within it
            # This might be complex; simpler to treat as deletes/creates for now
            logging.warning(f"Directory moved: {event.src_path} to {event.dest_path}. Re-indexing might be needed for contained notes.")
            # TODO: Potentially scan dest_path for new .md files if feasible
            return

        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)

        is_src_md = src_path.suffix.lower() == '.md' and self.vault_path in src_path.parents
        is_dest_md = dest_path.suffix.lower() == '.md' and self.vault_path in dest_path.parents

        if is_src_md:
            logging.info(f"Deleted (moved from): {event.src_path}")
            # TODO: Trigger remove file from index (using src_path)

        if is_dest_md:
            logging.info(f"Created (moved to): {event.dest_path}")
            # TODO: Trigger add file to index (using dest_path)

def start_watching(vault_path: str):
    """Starts the filesystem watcher."""
    event_handler = MarkdownEventHandler(vault_path)
    observer = Observer()
    observer.schedule(event_handler, vault_path, recursive=True)
    observer.start()
    logging.info(f"Started watching {vault_path} for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Watcher stopped by user.")
    observer.join()

# Example usage (for testing)
if __name__ == "__main__":
    # Replace with a valid path to your test vault
    # Ensure this path exists before running
    test_vault = "/path/to/your/obsidian/vault" # <<< CHANGE THIS
    if not Path(test_vault).is_dir():
         print(f"Error: Test vault path does not exist or is not a directory: {test_vault}")
         print("Please update the path in watcher.py before running.")
    else:
        start_watching(test_vault) 