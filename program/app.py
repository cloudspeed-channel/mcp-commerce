"""
shopping-agent/app.py  –  v2
======================
A Streamlit application acting as an MCP Client to search and scrape
e-commerce products using AWS Bedrock (Claude 3.5 Sonnet) as the AI brain,
Serper.dev for web search, and Firecrawl for deep page scraping.

Changes v2:
  • Product cards now show the product image (image_url field).
  • Individual product page URLs are extracted and linked (product_url).
  • Raw JSON from Claude's final reply is hidden inside a collapsible expander.
  • Product results from ALL past queries persist across new queries.

Architecture:
  User Query → Streamlit UI
              → AWS Bedrock (Claude 3.5 Sonnet) via converse()
              → MCP Tool Call (Serper search OR Firecrawl scrape)
              → MCP Server (stdio subprocess: uvx / npx)
              → Results returned to Claude → Final structured answer
              → Streamlit renders Product Cards
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library & third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import os
import re
from contextlib import AsyncExitStack

import boto3
import streamlit as st
from dotenv import load_dotenv

# MCP Python SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ──────────────────────────────────────────────────────────────────────────────
# Load .env (ignored if vars are already set in the environment)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# EC2 / Linux PATH fix
# Streamlit spawns with a stripped-down PATH. We inject the full system PATH so
# that `uvx` and `npx` (in /usr/local/bin or ~/.local/bin) can be found.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_ENV = {
    **os.environ,
    "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin"
             ":" + os.environ.get("PATH", ""),
}

# ──────────────────────────────────────────────────────────────────────────────
# AWS Bedrock model ID
# ──────────────────────────────────────────────────────────────────────────────
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Shopping Agent",
    page_icon="🛒",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS – product card styling
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 18px;
        transition: box-shadow 0.2s;
    }
    .product-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.13); }
    .product-card img {
        width: 100%;
        height: 230px;
        object-fit: cover;
        border-bottom: 1px solid #f0f0f0;
        display: block;
    }
    .no-image {
        width: 100%;
        height: 230px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f7f7f7;
        font-size: 3.5rem;
        border-bottom: 1px solid #ebebeb;
    }
    .card-body { padding: 14px 16px 8px 16px; }
    .card-name {
        font-size: 0.93rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .card-price {
        font-size: 1.2rem;
        font-weight: 800;
        color: #c0392b;
        margin-bottom: 5px;
    }
    .card-meta { font-size: 0.82rem; color: #555; margin-bottom: 3px; }
    .card-desc {
        font-size: 0.79rem;
        color: #777;
        margin-top: 7px;
        line-height: 1.45;
    }
    .card-btn {
        display: block;
        margin: 10px 16px 14px 16px;
        padding: 9px 0;
        text-align: center;
        background: #1a1a1a;
        color: #fff !important;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none !important;
    }
    .card-btn:hover { background: #333; }
    .query-badge {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Credentials & Configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("🔍 Serper (Search)")
    serper_key = st.text_input(
        "SERPER_API_KEY",
        value=os.getenv("SERPER_API_KEY", ""),
        type="password",
        help="Get your free key at https://serper.dev",
    )

    st.subheader("🕸️ Firecrawl (Scrape)")
    firecrawl_key = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Get your key at https://firecrawl.dev",
    )

    st.subheader("☁️ AWS Bedrock")
    aws_access_key = st.text_input(
        "AWS_ACCESS_KEY_ID",
        value=os.getenv("AWS_ACCESS_KEY_ID", ""),
        type="password",
    )
    aws_secret_key = st.text_input(
        "AWS_SECRET_ACCESS_KEY",
        value=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        type="password",
    )
    aws_region = st.text_input(
        "AWS_REGION",
        value=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    st.markdown("---")
    # Clear button – explicit user action to wipe results
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.all_search_results = []
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Keys are used only within this session and never stored.")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS – MCP ↔ Bedrock schema conversion
# ══════════════════════════════════════════════════════════════════════════════

def mcp_tool_to_bedrock_tool(mcp_tool) -> dict:
    """
    Convert a single MCP Tool object into the AWS Bedrock Converse API
    ToolSpec format.

    Bedrock Converse ToolSpec (DIFFERENT from OpenAI):
        {
          "toolSpec": {
            "name": "...",
            "description": "...",
            "inputSchema": {
              "json": { ...actual JSON Schema... }   ← nested under "json"
            }
          }
        }
    """
    raw_schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}
    return {
        "toolSpec": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            # ⚠️  Bedrock requires the JSON schema nested under the "json" key
            "inputSchema": {"json": raw_schema},
        }
    }


def build_bedrock_tool_config(mcp_tools: list) -> dict:
    """Build the full toolConfig dict for boto3 bedrock-runtime converse()."""
    return {"tools": [mcp_tool_to_bedrock_tool(t) for t in mcp_tools]}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – split human-readable text from embedded JSON block
# ══════════════════════════════════════════════════════════════════════════════

def split_text_and_json(text: str) -> tuple:
    """
    Separate the friendly summary from any ```json … ``` code block.

    Returns:
        (clean_text, raw_json_string | None)

    The clean_text has the JSON block stripped out so it doesn't clutter
    the chat UI. The raw JSON is shown in a separate collapsible expander.
    """
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match   = re.search(pattern, text, re.IGNORECASE)
    if match:
        clean = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return clean, match.group(1).strip()
    return text, None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – build one HTML product card
# ══════════════════════════════════════════════════════════════════════════════

def render_product_card_html(product: dict) -> str:
    """
    Return an HTML string for a single product card.

    Expected product keys:
        name, price, brand, color, description
        url          – the seller / listing page Claude scraped
        product_url  – direct link to the individual item (may equal url)
        image_url    – direct URL of the product's main image
    """
    name        = product.get("name", "Unknown Product")
    price       = product.get("price", "N/A")
    brand       = product.get("brand", "N/A")
    color       = product.get("color", "N/A")
    description = product.get("description", "")
    image_url   = (product.get("image_url") or "").strip()

    # Prefer a dedicated individual-item URL, fall back to the scraped page URL
    link_url = (
        product.get("product_url")
        or product.get("url")
        or ""
    ).strip()

    # Truncate long descriptions
    if len(description) > 170:
        description = description[:167] + "…"

    # ── Image section ──────────────────────────────────────────────────────────
    # onerror hides broken images gracefully
    if image_url:
        img_block = (
            f'<img src="{image_url}" alt="{name}" '
            f'onerror="this.parentNode.innerHTML=\'<div class=&quot;no-image&quot;>🛍️</div>\'">'
        )
    else:
        img_block = '<div class="no-image">🛍️</div>'

    # ── Button ─────────────────────────────────────────────────────────────────
    btn_block = (
        f'<a class="card-btn" href="{link_url}" target="_blank" rel="noopener">'
        f'🔗 View Product</a>'
        if link_url else ""
    )

    desc_block = (
        f'<div class="card-desc">{description}</div>' if description else ""
    )

    return f"""
<div class="product-card">
    {img_block}
    <div class="card-body">
        <div class="card-name">{name}</div>
        <div class="card-price">{price}</div>
        <div class="card-meta">🏷️ <b>Brand:</b> {brand}</div>
        <div class="card-meta">🎨 <b>Color:</b> {color}</div>
        {desc_block}
    </div>
    {btn_block}
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE ASYNC LOGIC – MCP sessions + Bedrock agentic loop
# ══════════════════════════════════════════════════════════════════════════════

async def run_shopping_agent(
    user_query: str,
    serper_api_key: str,
    firecrawl_api_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str,
    status_callback,
) -> dict:
    """
    Core async agent that:
      1. Starts Serper + Firecrawl MCP servers over stdio.
      2. Discovers their tools and converts them to Bedrock format.
      3. Runs a multi-turn converse() loop until Claude finishes.
      4. Returns {"messages": [...], "products": [...]}
    """

    # ── Inject keys into subprocess env ───────────────────────────────────────
    env_with_keys = {
        **SYSTEM_ENV,
        "SERPER_API_KEY": serper_api_key,
        "FIRECRAWL_API_KEY": firecrawl_api_key,
    }

    # ── MCP server stdio parameters ────────────────────────────────────────────
    serper_params = StdioServerParameters(
        command="uvx",
        args=["serper-mcp-server"],
        env=env_with_keys,
    )
    firecrawl_params = StdioServerParameters(
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env=env_with_keys,
    )

    # ── Bedrock boto3 client ───────────────────────────────────────────────────
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    conversation_log = []
    products         = []

    # AsyncExitStack manages both MCP server lifecycles together
    async with AsyncExitStack() as stack:

        # ── Start Serper ───────────────────────────────────────────────────────
        status_callback("🔌 Connecting to Serper MCP server…")
        s_read, s_write = await stack.enter_async_context(
            stdio_client(serper_params)
        )
        serper_session: ClientSession = await stack.enter_async_context(
            ClientSession(s_read, s_write)
        )
        await serper_session.initialize()

        # ── Start Firecrawl ────────────────────────────────────────────────────
        status_callback("🔌 Connecting to Firecrawl MCP server…")
        f_read, f_write = await stack.enter_async_context(
            stdio_client(firecrawl_params)
        )
        firecrawl_session: ClientSession = await stack.enter_async_context(
            ClientSession(f_read, f_write)
        )
        await firecrawl_session.initialize()

        # ── Discover tools from both servers ──────────────────────────────────
        status_callback("🔧 Discovering MCP tools…")
        serper_tools    = (await serper_session.list_tools()).tools
        firecrawl_tools = (await firecrawl_session.list_tools()).tools
        all_mcp_tools   = serper_tools + firecrawl_tools

        # Map tool name → owning session for fast routing
        tool_to_session: dict = {
            **{t.name: serper_session    for t in serper_tools},
            **{t.name: firecrawl_session for t in firecrawl_tools},
        }

        bedrock_tool_config = build_bedrock_tool_config(all_mcp_tools)

        # ── System prompt ──────────────────────────────────────────────────────
        # Explicitly requests image_url and product_url from Claude
        system_prompt = [
            {
                "text": (
                    "You are an expert e-commerce shopping assistant.\n"
                    "When the user asks about products follow these steps:\n\n"
                    "STEP 1 – SEARCH: Use the search tool to find e-commerce "
                    "product pages (Amazon, eBay, brand sites, etc). "
                    "Pick the 2-3 most relevant result URLs.\n\n"
                    "STEP 2 – SCRAPE: Use the scrape/crawl tool on each URL. "
                    "Extract:\n"
                    "  • name – product title\n"
                    "  • price – include currency symbol\n"
                    "  • brand\n"
                    "  • color / variant\n"
                    "  • product_url – the direct link to that INDIVIDUAL item "
                    "    (not a category or search-results page; look for the "
                    "    canonical URL, the href on the product title, or the "
                    "    'Buy now' button link)\n"
                    "  • image_url – the main product image URL; check "
                    "    og:image meta tag first, then large <img> src "
                    "    attributes, then CDN image URLs\n"
                    "  • description – a short sentence about the product\n\n"
                    "STEP 3 – REPLY: Write 1-2 friendly sentences summarising "
                    "what you found. Do NOT list the products in plain text. "
                    "Then output exactly ONE ```json code block containing a "
                    "JSON array. Each element must have these keys:\n"
                    "  name, price, brand, color, url, product_url, "
                    "  image_url, description\n\n"
                    "Put the ```json block AFTER your summary. "
                    "Do not repeat the product data outside the JSON block."
                )
            }
        ]

        # ── Seed the conversation ──────────────────────────────────────────────
        messages = [{"role": "user", "content": [{"text": user_query}]}]
        conversation_log.append({"role": "user", "content": user_query})

        # ══════════════════════════════════════════════════════════════════════
        # AGENTIC LOOP – runs until Claude issues end_turn with no tool calls
        # ══════════════════════════════════════════════════════════════════════
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            status_callback(f"🤖 Thinking… (turn {iteration})")

            response = bedrock.converse(
                modelId=BEDROCK_MODEL_ID,
                system=system_prompt,
                messages=messages,
                toolConfig=bedrock_tool_config,
            )

            stop_reason    = response.get("stopReason", "")
            output_message = response["output"]["message"]

            # Always append the assistant message back to history
            messages.append(output_message)

            # Separate text blocks from tool-use requests
            text_parts      = []
            tool_use_blocks = []
            for block in output_message.get("content", []):
                if "text"    in block: text_parts.append(block["text"])
                if "toolUse" in block: tool_use_blocks.append(block["toolUse"])

            if text_parts:
                conversation_log.append(
                    {"role": "assistant", "content": "\n".join(text_parts)}
                )

            # ── Done? ──────────────────────────────────────────────────────────
            if stop_reason == "end_turn" and not tool_use_blocks:
                status_callback("✅ Done!")
                break

            # ── Execute every tool call Claude requested ───────────────────────
            if tool_use_blocks:
                tool_results_content = []

                for tu in tool_use_blocks:
                    tool_name   = tu["name"]
                    tool_input  = tu["input"]
                    tool_use_id = tu["toolUseId"]

                    status_callback(f"🔧 Running tool `{tool_name}`…")
                    conversation_log.append(
                        {
                            "role": "tool_call",
                            "content": (
                                f"**Tool:** `{tool_name}`\n"
                                f"**Input:**\n```json\n"
                                f"{json.dumps(tool_input, indent=2)}\n```"
                            ),
                        }
                    )

                    # Route to the correct MCP session
                    session = tool_to_session.get(tool_name)
                    if session is None:
                        result_text = f"Error: unknown tool '{tool_name}'"
                    else:
                        try:
                            mcp_result  = await session.call_tool(
                                tool_name, arguments=tool_input
                            )
                            result_text = "\n".join(
                                p.text if hasattr(p, "text") else str(p)
                                for p in mcp_result.content
                            )
                        except Exception as exc:
                            result_text = f"Tool error: {exc}"

                    # Truncate to avoid blowing out the context window
                    MAX_CHARS = 14_000
                    if len(result_text) > MAX_CHARS:
                        result_text = result_text[:MAX_CHARS] + "\n…[truncated]"

                    conversation_log.append(
                        {"role": "tool_result", "content": result_text}
                    )
                    tool_results_content.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": result_text}],
                            }
                        }
                    )

                # Feed results back as a user turn (Bedrock requirement)
                messages.append({"role": "user", "content": tool_results_content})
                continue  # back to top of loop

            break  # no tool calls, no end_turn → bail

        # ══════════════════════════════════════════════════════════════════════
        # Parse products out of Claude's last assistant message
        # ══════════════════════════════════════════════════════════════════════
        final_text = next(
            (m["content"] for m in reversed(conversation_log)
             if m["role"] == "assistant"),
            ""
        )
        _, raw_json = split_text_and_json(final_text)
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    products = parsed
                elif isinstance(parsed, dict) and "products" in parsed:
                    products = parsed["products"]
            except json.JSONDecodeError:
                products = []

    return {"messages": conversation_log, "products": products}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# all_search_results: list of {"query": str, "products": list}
# Results ACCUMULATE – new queries append; nothing is auto-wiped.
if "all_search_results" not in st.session_state:
    st.session_state.all_search_results = []

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛒 AI Shopping Agent")
st.caption(
    "Powered by AWS Bedrock (Claude 3.5 Sonnet) · Serper MCP · Firecrawl MCP"
)


# ══════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_history():
    """
    Render the full conversation log.
    - Assistant messages: friendly summary is shown; JSON block is hidden in
      a collapsible expander so the user can inspect it but it never clutters
      the conversation.
    - Tool calls / results: collapsed by default.
    """
    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                clean_text, raw_json = split_text_and_json(content)
                # Show the human-readable summary
                if clean_text:
                    st.markdown(clean_text)
                # Raw JSON goes into a dropdown – never auto-expanded
                if raw_json:
                    with st.expander("📋 View raw product JSON"):
                        st.code(raw_json, language="json")

        elif role == "tool_call":
            with st.chat_message("assistant", avatar="🔧"):
                with st.expander("🔧 Tool call — click to expand"):
                    st.markdown(content)

        elif role == "tool_result":
            with st.chat_message("assistant", avatar="📦"):
                with st.expander("📦 Tool result — click to expand"):
                    st.code(content, language="text")


def render_all_product_sections():
    """
    Render a product card grid for EVERY past query.
    Each section is headed by a badge showing the original search query.
    Sections are never removed when new queries arrive – the user sees a
    running history of all their searches.
    """
    for entry in st.session_state.all_search_results:
        query    = entry["query"]
        products = entry["products"]

        if not products:
            st.markdown("---")
            st.markdown(
                f'<span class="query-badge">🔍 {query}</span>',
                unsafe_allow_html=True,
            )
            st.info("No products could be extracted for this query.")
            continue

        st.markdown("---")
        st.markdown(
            f'<span class="query-badge">🔍 {query}</span>',
            unsafe_allow_html=True,
        )
        st.subheader("🛍️ Products Found")

        # 3-column responsive grid
        cols_per_row = 3
        rows = [
            products[i : i + cols_per_row]
            for i in range(0, len(products), cols_per_row)
        ]
        for row in rows:
            cols = st.columns(len(row))
            for col, product in zip(cols, row):
                with col:
                    st.markdown(
                        render_product_card_html(product),
                        unsafe_allow_html=True,
                    )


# Render existing history on every page load / rerun
render_chat_history()
render_all_product_sections()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT – new query
# ══════════════════════════════════════════════════════════════════════════════
if prompt := st.chat_input(
    "e.g. Find me blue running shoes under $100 on Amazon…"
):
    # ── Validate credentials ──────────────────────────────────────────────────
    missing = [
        k for k, v in {
            "SERPER_API_KEY":        serper_key,
            "FIRECRAWL_API_KEY":     firecrawl_key,
            "AWS_ACCESS_KEY_ID":     aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_key,
        }.items()
        if not v
    ]
    if missing:
        st.error(
            f"⚠️ Please fill in the sidebar credentials: {', '.join(missing)}"
        )
        st.stop()

    # ── Append user message and show it immediately ───────────────────────────
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Live status placeholder ───────────────────────────────────────────────
    status_box = st.empty()

    def update_status(msg: str):
        status_box.info(msg)

    # ── Run the async agent ───────────────────────────────────────────────────
    with st.spinner("🤖 Agent is working…"):
        result = asyncio.run(
            run_shopping_agent(
                user_query=prompt,
                serper_api_key=serper_key,
                firecrawl_api_key=firecrawl_key,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_region=aws_region,
                status_callback=update_status,
            )
        )

    status_box.empty()

    # ── Persist new conversation messages ────────────────────────────────────
    # The initial user message was already appended above, so skip it here.
    for msg in result["messages"]:
        if msg["role"] == "user" and msg["content"] == prompt:
            continue
        st.session_state.chat_history.append(msg)

    # ── Append this query's products to the PERSISTENT results list ───────────
    # We always push a new entry so the query badge is shown even if 0 products.
    # Products from PREVIOUS queries are never removed.
    st.session_state.all_search_results.append(
        {"query": prompt, "products": result["products"]}
    )

    # ── Re-render the full page with updated state ────────────────────────────
    st.rerun()