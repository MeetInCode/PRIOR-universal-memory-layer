from prior_memory_layer import PriorMemoryLayer
from rich.console import Console
from dotenv import load_dotenv
import sys

load_dotenv()
console = Console()

def test_connection():
    console.print("[bold cyan]Testing PriorMemoryLayer Connection...[/]")
    try:
        layer = PriorMemoryLayer()
        console.print("[green]PASS: PriorMemoryLayer instantiated successfully.[/]")
        
        # Test Neo4j connectivity
        res = layer.graph.query("RETURN 1 as val")
        if res and res[0]['val'] == 1:
            console.print("[green]PASS: Neo4j Connection verified.[/]")
        else:
            console.print("[red]FAIL: Neo4j Query returned unexpected result.[/]")
            
    except Exception as e:
        console.print(f"[red]FAIL: {e}[/]")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
