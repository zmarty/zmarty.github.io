---
layout: post
tags: [gpt-oss, llm]
---

## Harnessing gpt-oss built-in tools

The [OpenAI gpt-oss models](https://openai.com/index/introducing-gpt-oss/) come with built-in tools (python and browser) that are deeply integrated into the model's training. Since these tools are built-in, the inference engine itself must handle them - not your application code.

This blog post explains how to properly set up vLLM with these tools and integrate it with LibreChat to leverage these powerful capabilities.

### Built-in tool basics

The gpt-oss series are OpenAI's open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases. They are available in two flavors: gpt-oss-120b (117B parameters) for production use and gpt-oss-20b (21B parameters) for lower latency applications.

The models come with two built-in tools that are deeply integrated into their training:

- **python tool**: Executes Python code in a stateful Jupyter environment for calculations, data analysis, and generating visualizations. Code runs in the model's chain of thought and isn't shown to users unless explicitly intended.
- **browser tool**: Searches for information, opens web pages, and finds specific content. Includes functions for `search`, `open`, and `find` operations with proper citation formatting.

Using these built-in tools is advantageous because they're trained directly into the model's behavior, use the proper `analysis` channel for seamless reasoning integration, and have been optimized for accuracy and reliability compared to custom functions that would duplicate this functionality. This is because gpt-oss uses the [Harmony response format](https://github.com/openai/harmony) with three distinct channels, and built-in tools are trained to output to a specific channel. The inference engine must parse these channels, route messages appropriately, and filter content correctly:

- **`analysis`**: Contains the model's chain of thought (CoT) reasoning. This channel should not be shown to end users as it doesn't adhere to the same safety standards as final output.
- **`commentary`**: Used for custom function tool calls and occasional preambles when calling multiple functions.
- **`final`**: User-facing messages that represent the actual responses intended for end users.

## Install vLLM

For my setup, I run the gpt-oss-20b model on 2× NVIDIA RTX 3090 GPUs (24GB VRAM each), which provides good performance for the smaller variant while maintaining reasonable inference speeds for interactive applications.

Create a new Python environment:

```bash
uv venv venv --python 3.12 --seed
source venv/bin/activate
```

Install vLLM with automatic torch backend detection:

```bash
uv pip install vllm --torch-backend=auto
```

## Start the browser MCP server

OpenAI provides reference implementations for the built-in tools through the [gpt-oss package](https://pypi.org/project/gpt-oss/). To make the browser tool available to vLLM, we'll create an MCP (Model Context Protocol) server that wraps the reference implementation.

<details markdown="1">
<summary markdown="span">Create a file named `browser_server.py` (click to expand)</summary>

```python
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Union, Optional

from mcp.server.fastmcp import Context, FastMCP
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend, ExaBackend

@dataclass
class AppContext:
    browsers: dict[str, SimpleBrowserTool] = field(default_factory=dict)

    def create_or_get_browser(self, session_id: str) -> SimpleBrowserTool:
        if session_id not in self.browsers:
            tool_backend = os.getenv("BROWSER_BACKEND", "exa")
            if tool_backend == "youcom":
                backend = YouComBackend(source="web")
            elif tool_backend == "exa":
                backend = ExaBackend(source="web")
            else:
                raise ValueError(f"Invalid tool backend: {tool_backend}")
            self.browsers[session_id] = SimpleBrowserTool(backend=backend)
        return self.browsers[session_id]

    def remove_browser(self, session_id: str) -> None:
        self.browsers.pop(session_id, None)


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    yield AppContext()


# Pass lifespan to server
mcp = FastMCP(
    name="browser",
    instructions=r"""
Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`. 
Do not quote more than 10 words directly from the tool output.
sources=web
""".strip(),
    lifespan=app_lifespan,
    port=8001,
)


@mcp.tool(
    name="search",
    title="Search for information",
    description=
    "Searches for information related to `query` and displays `topn` results.",
)
async def search(ctx: Context,
                 query: str,
                 topn: int = 10,
                 source: Optional[str] = None) -> str:
    """Search for information related to a query"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.search(query=query, topn=topn, source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)


@mcp.tool(
    name="open",
    title="Open a link or page",
    description="""
Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
Valid link ids are displayed with the formatting: `【{id}†.*】`.
If `cursor` is not provided, the most recent page is implied.
If `id` is a string, it is treated as a fully qualified URL associated with `source`.
If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
Use this function without `id` to scroll to a new location of an opened page.
""".strip(),
)
async def open_link(ctx: Context,
                    id: Union[int, str] = -1,
                    cursor: int = -1,
                    loc: int = -1,
                    num_lines: int = -1,
                    view_source: bool = False,
                    source: Optional[str] = None) -> str:
    """Open a link or navigate to a page location"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.open(id=id,
                                      cursor=cursor,
                                      loc=loc,
                                      num_lines=num_lines,
                                      view_source=view_source,
                                      source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)


@mcp.tool(
    name="find",
    title="Find pattern in page",
    description=
    "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
)
async def find_pattern(ctx: Context, pattern: str, cursor: int = -1) -> str:
    """Find exact matches of a pattern in the current page"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.find(pattern=pattern, cursor=cursor):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)
```

</details>

Before you start the server, you need to get an API key from [exa.ai](https://exa.ai/exa-api), which the server uses to browse the web.

Start the MCP server using the `fastmcp` CLI:

```bash
export EXA_API_KEY=YOUR-EXA-KEY-HERE
mcp run -t sse browser_server.py:mcp
```

This starts the browser tool server on port 8001. You should see output similar to:

```text
INFO:     Started server process [730909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

---

## Start vLLM with browser tool integration

Now we can launch vLLM and configure it to use the browser MCP server we just started. The key parameter is `--tool-server` which points to our MCP server on `localhost:8001`.

```bash
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --max_num_seqs 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-server localhost:8001
```

Once vLLM initializes, you should see output indicating the server is running:

```text
(APIServer pid=732684) INFO:     Started server process [732684]
(APIServer pid=732684) INFO:     Waiting for application startup.
(APIServer pid=732684) INFO:     Application startup complete.
```

## Test from python

<details markdown="1">
<summary markdown="span">Python test code using the OpenAI SDK (click to expand)</summary>

```python
from openai import OpenAI
import json
from pprint import pprint
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich import box


def display_response_flow(response):
    """
    Display the response in a nice, structured format showing:
    - Each reasoning step
    - Each tool call with its action
    - The final assistant message
    - Token usage statistics
    """
    console = Console()
    
    # Header with status
    status_color = {
        'completed': 'green',
        'in_progress': 'yellow',
        'failed': 'red',
        'cancelled': 'red'
    }.get(response.status, 'cyan')
    
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Response Flow: {response.model}[/bold cyan]\n"
        f"[bold]Status:[/bold] [{status_color}]{response.status}[/{status_color}]",
        border_style="cyan"
    ))
    console.print()
    
    if not hasattr(response, 'output') or not response.output:
        console.print("[yellow]No output in response[/yellow]")
        return
    
    step_num = 1
    reasoning_count = 0
    tool_call_count = 0
    has_final_message = False
    
    for output_item in response.output:
        # Display reasoning blocks
        if output_item.type == 'reasoning':
            reasoning_count += 1
            for content in output_item.content:
                if content.type == 'reasoning_text':
                    console.print(Panel(
                        Text(content.text, style="italic dim"),
                        title=f"[bold yellow]💭 Reasoning #{reasoning_count}[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                        padding=(1, 2)
                    ))
                    console.print()
        
        # Display tool calls
        elif output_item.type == 'web_search_call':
            tool_call_count += 1
            action = output_item.action
            status = output_item.status or "unknown"
            
            # Format action details - action is a Pydantic model, not a dict
            action_type = getattr(action, 'type', 'N/A')
            action_text = f"[bold]Type:[/bold] {action_type}\n"
            
            if hasattr(action, 'query') and action.query:
                action_text += f"[bold]Query:[/bold] {action.query}\n"
            if hasattr(action, 'url') and action.url:
                action_text += f"[bold]URL:[/bold] {action.url}\n"
            if hasattr(action, 'pattern') and action.pattern:
                action_text += f"[bold]Pattern:[/bold] {action.pattern}\n"
            
            action_text += f"[bold]Status:[/bold] {status}"
            
            # Choose icon based on action type
            icon = {
                'search': '🔍',
                'open_page': '📄',
                'find': '🔎',
            }.get(action_type, '🔧')
            
            console.print(Panel(
                action_text,
                title=f"[bold green]{icon} Tool Call #{tool_call_count}[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2)
            ))
            console.print()
        
        # Display final message
        elif output_item.type == 'message' and output_item.role == 'assistant':
            has_final_message = True
            for content in output_item.content:
                if content.type == 'output_text':
                    console.print(Panel(
                        Text(content.text),
                        title="[bold blue]📨 Final Response[/bold blue]",
                        border_style="blue",
                        box=box.DOUBLE,
                        padding=(1, 2)
                    ))
                    console.print()
    
    # Warning if no final message
    if not has_final_message:
        console.print(Panel(
            "[bold yellow]⚠️  No final response message found![/bold yellow]\n"
            "The model may have been cut off or encountered an error.",
            border_style="yellow",
            box=box.ROUNDED
        ))
        console.print()
    
    # Usage statistics
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        
        stats_text = f"""[bold]Input tokens:[/bold] {usage.input_tokens:,}
[bold]Output tokens:[/bold] {usage.output_tokens:,}
[bold]Total tokens:[/bold] {usage.total_tokens:,}"""
        
        if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
            if hasattr(usage.input_tokens_details, 'cached_tokens'):
                cached = usage.input_tokens_details.cached_tokens
                stats_text += f"\n[bold]Cached tokens:[/bold] {cached:,} ({cached/usage.input_tokens*100:.1f}%)"
        
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            details = usage.output_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                stats_text += f"\n[bold]Reasoning tokens:[/bold] {details.reasoning_tokens:,}"
            if hasattr(details, 'tool_output_tokens'):
                stats_text += f"\n[bold]Tool output tokens:[/bold] {details.tool_output_tokens:,}"
        
        console.print(Panel(
            stats_text,
            title="[bold magenta]📊 Token Usage[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2)
        ))
        console.print()
    
    # Summary
    console.print(Panel.fit(
        f"[bold]Summary:[/bold] {reasoning_count} reasoning steps, {tool_call_count} tool calls",
        border_style="cyan"
    ))
    console.print()


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
 
response = client.responses.create(
    model="openai/gpt-oss-20b",
    input="How is the weather in Seattle, WA?",
    tools=[
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        },
        {
            "type": "web_search_preview"
        }
    ],
    reasoning={
        "effort": "medium", # "low", "medium", or "high"
        "summary": "detailed"  # "auto", "concise", or "detailed"
    },
    temperature=1.0,
)

# Display the response in a nice format
display_response_flow(response)

# Show raw response for debugging if response looks incomplete
show_raw_debug = response.status != 'completed'
if show_raw_debug:  # Set to False to hide raw response
    print("\n" + "=" * 80)
    print("FULL RAW RESPONSE (for debugging)")
    print("=" * 80)
    response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
    pprint(response_dict, width=120, depth=10)
```

</details>

The crucial part of the code is telling vLLM to use the built-in tools. Note how we just reference the tool name without specifying an actual implementation, and vLLM internally properly calls the browser tool.

```python
tools=[
    {
        "type": "code_interpreter",
        "container": {
            "type": "auto"
        }
    },
    {
        "type": "web_search_preview"
    }
]
```