from prior_memory_layer import PriorMemoryLayer
from rich.console import Console
from dotenv import load_dotenv
import sys

load_dotenv()
console = Console()

def test_full_flow():
    console.print("[bold cyan]Testing PriorMemoryLayer Full Flow...[/]")
    try:
        layer = PriorMemoryLayer()
        layer.set_user_session("test_user@example.com")
        
        console.print("[dim]Sending test message to trigger GraphArchitect...[/]")
        # This will trigger classify -> architect -> execute
        # We process a message that SHOULD be ingested
        msg = "I am working on a new Python project called Cortex."
        console.print(f"Input: '{msg}'")
        
        # We can peek at the classifier directly if we want, but process_message prints to console too.
        layer.process_message(msg)
        
        console.print("[green]PASS: Processed message without error.[/]")
            
    except Exception as e:
        console.print(f"[red]FAIL: {e}[/]")
        sys.exit(1)

if __name__ == "__main__":
    test_full_flow()
