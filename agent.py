#!/usr/bin/env python3
"""
mac code — claude code for your Mac
"""

import json, sys, os, time, subprocess, re, threading, queue, difflib
import urllib.request, random
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding
from rich.columns import Columns

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")

# ── Self-improvement: failure logging ─────────────
LOGS_DIR = Path.home() / ".mac-code" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def log_interaction(query, intent, response, speed, grade=None, error=None):
    """Log every interaction for self-improvement training data."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "intent": intent,
        "response": response[:500] if response else None,
        "speed": speed,
        "grade": grade,  # "good", "bad", or None (ungraded)
        "error": error,
        "model": get_current_model() if 'get_current_model' in dir() else "unknown",
    }
    log_file = LOGS_DIR / f"interactions-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_failure_stats():
    """Show stats from logged interactions."""
    total = 0
    graded = {"good": 0, "bad": 0}
    intents = {"search": 0, "shell": 0, "code": 0, "chat": 0}
    errors = 0

    for log_file in LOGS_DIR.glob("interactions-*.jsonl"):
        for line in open(log_file):
            try:
                entry = json.loads(line)
                total += 1
                if entry.get("grade"):
                    graded[entry["grade"]] = graded.get(entry["grade"], 0) + 1
                if entry.get("intent"):
                    intents[entry["intent"]] = intents.get(entry["intent"], 0) + 1
                if entry.get("error"):
                    errors += 1
            except:
                pass

    return {"total": total, "graded": graded, "intents": intents, "errors": errors}
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── model configs ─────────────────────────────────
MODELS = {
    "9b": {
        "path": os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf"),
        "ctx": 32768,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -t 4",
        "name": "Qwen3.5-9B",
        "detail": "8.95B dense · Q4_K_M · 32K ctx",
        "good_for": "tool calling, long conversations, agent tasks",
    },
    "35b": {
        "path": os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"),
        "ctx": 8192,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -np 1 -t 4",
        "name": "Qwen3.5-35B-A3B",
        "detail": "MoE 34.7B · 3B active · IQ2_M · 8K ctx",
        "good_for": "reasoning, math, knowledge, fast answers",
    },
}

# ── smart routing ─────────────────────────────────
TOOL_KEYWORDS = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "when does", "when are", "who do", "who is playing",
    "who plays", "who won", "what happened", "what is the score",
    "weather", "news", "latest", "schedule", "score", "tonight",
    "today", "tomorrow", "yesterday", "this week", "next game",
    "play next", "playing next", "results", "standings",
    "price", "stock", "market", "crypto", "bitcoin",
    "fetch", "download", "read file", "write file",
    "create file", "run", "execute", "list files", "show me",
    "open", "browse", "url", "http", "website",
    "how much", "where is", "directions", "recipe",
    "explore", "repo", "repository", "github", "tell me more",
    "more about", "what else", "continue", "go deeper",
]

VALID_INTENTS = ("search", "code", "shell", "chat")

# Sent on every conversational turn. Without it the model gets a bare user
# message, falls back to its base-instruct persona, and claims it cannot edit
# files — which is exactly the opposite of what this agent can do.
DEFAULT_SYSTEM_PROMPT = """You are mac code, a coding agent running on the user's own Mac in {work_dir}.

You DO have file tools. You can read, create, and edit files in that directory. When the user asks for a change, the request is routed to your tools automatically: you read the file, produce the edit, and the user sees a diff and approves it before anything touches disk.

So never say you cannot edit files, that you lack permission, or that you are read-only. You can. If someone asks whether you can make a change, answer yes and ask what specifically to change.

You are answering in conversation mode right now, so keep it short and concrete. When the user wants a file actually changed, name the change you would make and ask for the specifics — the edit itself runs through your tools, not through this reply."""

def with_system(messages, work_dir="."):
    """Prepend the default system prompt unless /system already set one."""
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system",
             "content": DEFAULT_SYSTEM_PROMPT.format(work_dir=work_dir)}] + messages

# Deterministic safety net for routing. A mis-routed edit request fails
# silently — the model just answers in chat and never touches a file — so
# catch unambiguous edit requests before that happens.
EDIT_VERB_RE = re.compile(
    r"\b(add|append|create|write|implement|refactor|rename|delete|remove|"
    r"fix|repair|update|change|edit|modify|replace|rewrite|convert|port|"
    r"introduce|extract|generate|insert|patch|reorder|comment|annotate)\b", re.I)
CODE_TARGET_RE = re.compile(
    r"\b\w+\.(py|js|jsx|ts|tsx|json|md|txt|sh|bash|zsh|go|rs|java|rb|php|"
    r"c|h|cpp|hpp|cs|swift|kt|sql|html|css|yml|yaml|toml|ini)\b|"
    r"\b(function|method|class|module|script|import|variable|constant|"
    r"line \d+|def |bug|error|exception|traceback|refactor|codebase|repo)\b", re.I)
# Questions about what you can do, or asking for an explanation, are not edits.
META_QUESTION_RE = re.compile(
    r"^\s*(are|is|can|could|do|does|did|would|will|should|what|why|how|who|"
    r"when|where|which|explain|describe|tell me about)\b", re.I)

def looks_like_edit_request(message):
    if META_QUESTION_RE.match(message):
        return False
    return bool(EDIT_VERB_RE.search(message) and CODE_TARGET_RE.search(message))

def classify_intent(message):
    """Ask LLM to classify: 'search', 'shell', 'code', or 'chat'. One fast call (~1s)."""
    try:
        result, _ = llm_call([
            {"role": "system", "content": """Classify the user's request into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- search: needs web search (news, scores, weather, prices, current events, looking up info online)
- code: wants a file on this computer created or CHANGED. Any request to edit, fix, refactor, rename, add to, or write a file, script, or function. Also "why does X crash" when X is a file in the working directory.
- shell: needs to inspect the computer without changing anything (find files, list directories, read a file, check disk space, what is running on a port)
- chat: general conversation, reasoning, math, and questions ABOUT code that do not require touching a file ("explain this regex", "what does this function do", "write me a snippet to try")

The line between code and chat is whether a file gets modified. Explaining is chat. Changing is code.

Reply with ONLY one word: search, code, shell, or chat"""},
            {"role": "user", "content": message},
        ], max_tokens=5, temperature=0.0)
        # Models like to answer "code." or "Category: code" — scan every token
        # for a known label, and fall back to chat on anything unrecognised.
        for tok in re.findall(r"[a-z]+", (result or "").lower()):
            if tok in VALID_INTENTS:
                return tok
        return "chat"
    except Exception:
        return "chat"

def generate_shell_command(query, work_dir="."):
    """Ask LLM to generate the right shell command for a file/system task."""
    home = os.path.expanduser("~")
    result, _ = llm_call([
        {"role": "system", "content": f"""You are a macOS shell command generator. The user's home directory is {home}. Current working directory is {work_dir}.

Generate a single shell command that accomplishes the user's request. Output ONLY the command, nothing else. No explanation, no markdown, no backticks.

Examples:
- "find videos on my desktop" → find {home}/Desktop -type f \\( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \\)
- "what files are on my desktop" → ls -la {home}/Desktop
- "how much disk space do I have" → df -h /
- "show me python files in this project" → find . -name "*.py" -type f
- "read the readme" → cat README.md
- "what's running on port 8000" → lsof -i :8000
- "count lines of code" → find . -name "*.py" -exec wc -l {{}} +"""},
        {"role": "user", "content": query},
    ], max_tokens=100, temperature=0.0)
    return result.strip().strip('`').strip()

def run_smart_tool(query, work_dir="."):
    """Execute a shell command generated by the LLM, feed results back."""
    import subprocess as sp
    from datetime import datetime

    # Step 1: LLM generates the command (~1s)
    cmd = generate_shell_command(query, work_dir)

    # Step 2: Execute it
    try:
        result = sp.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=30, cwd=work_dir)
        output = result.stdout[:8000]
        if result.stderr:
            output += f"\n{result.stderr[:2000]}"
    except sp.TimeoutExpired:
        output = "Command timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {e}"

    # Step 3: LLM summarizes results (~2-3s)
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. You ran a shell command and got results. Present the results clearly to the user. If it's a file listing, format it nicely. If it's code, use formatting. Be helpful and concise."},
        {"role": "user", "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0), cmd

def run_file_tool(query, work_dir="."):
    """Execute file/exec operations directly in Python, feed results to LLM."""
    import subprocess as sp
    from datetime import datetime

    lower = query.lower()
    tool_output = ""
    tool_name = ""

    try:
        # List directory
        if any(kw in lower for kw in ["list files", "list dir", "ls ", "what's in"]):
            # Extract path or use work_dir
            path = work_dir
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isdir(expanded):
                    path = expanded
                    break
            entries = os.listdir(path)
            entries.sort()
            tool_name = f"list_dir({path})"
            tool_output = "\n".join(entries[:50])
            if len(entries) > 50:
                tool_output += f"\n... and {len(entries)-50} more"

        # Read file
        elif any(kw in lower for kw in ["read file", "show me", "look at", "cat ", "what's in"]):
            # Find file path in the query
            path = None
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isfile(expanded):
                    path = expanded
                    break
                # Try with work_dir
                joined = os.path.join(work_dir, token)
                if os.path.isfile(joined):
                    path = joined
                    break
            if path:
                with open(path, "r", errors="ignore") as f:
                    content = f.read(10000)
                tool_name = f"read_file({path})"
                tool_output = content
            else:
                tool_output = f"Could not find file in query: {query}"
                tool_name = "read_file(not found)"

        # Write file
        elif any(kw in lower for kw in ["write file", "write a file", "create file", "create a file",
                                          "create a new", "save file", "save to", "save this"]):
            # LLM decides what to write
            content, _ = llm_call([
                {"role": "system", "content": "The user wants to create/write a file. Generate ONLY the file content. No explanations."},
                {"role": "user", "content": query},
            ], max_tokens=2000)

            # Extract filename from query
            filename = None
            for token in query.split():
                if "." in token and not token.startswith("http"):
                    filename = token
                    break
            if not filename:
                filename = "output.txt"

            filepath = os.path.join(work_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            tool_name = f"write_file({filepath})"
            tool_output = f"Written {len(content)} bytes to {filepath}"

        # Execute command
        elif any(kw in lower for kw in ["execute", "run "]):
            # Extract command
            cmd = query
            for prefix in ["execute ", "run "]:
                if lower.startswith(prefix):
                    cmd = query[len(prefix):]
                    break

            result = sp.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=30, cwd=work_dir)
            tool_name = f"exec({cmd.strip()[:40]})"
            tool_output = result.stdout[:5000]
            if result.stderr:
                tool_output += f"\nSTDERR: {result.stderr[:1000]}"

        else:
            return None

    except Exception as e:
        tool_output = f"Error: {e}"
        tool_name = "error"

    # Feed tool output to LLM for final answer
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. You executed a tool and got results. Summarize the results clearly for the user. If it's code, format it nicely."},
        {"role": "user", "content": f"Tool: {tool_name}\nResult:\n{tool_output}\n\nOriginal question: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0), tool_name

def llm_call(messages, max_tokens=300, temperature=0.1):
    """Single LLM call, returns content + timings."""
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return d["choices"][0]["message"]["content"], d.get("timings", {})

def quick_search(query):
    """LLM rewrites query → DuckDuckGo search → LLM answers. ~5-8s total."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")

    # Step 1: LLM rewrites query into optimal search terms (~1s)
    try:
        search_query, _ = llm_call([
            {"role": "system", "content": f"Today is {today}. Rewrite the user's question into an optimal web search query that will find current, specific data (not articles about announcements). Include 'today' or 'tonight' and the full date for time-sensitive queries. Add words like 'scores', 'results', 'live', or 'now' when looking for current data. Output ONLY the search query string, nothing else."},
            {"role": "user", "content": query},
        ], max_tokens=30, temperature=0.0)
        search_query = search_query.strip().strip('"\'')
    except Exception:
        search_query = query

    # Step 2: DuckDuckGo search — text (15 results) + news (5 results)
    ddg = DDGS()
    all_results = []

    try:
        text_results = ddg.text(search_query, max_results=15)
        all_results.extend(text_results)
    except Exception:
        pass

    try:
        news_results = ddg.news(search_query, max_results=5)
        all_results.extend(news_results)
    except Exception:
        pass

    if not all_results:
        return None

    # Combine all snippets
    snippets = "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in all_results])

    # Check if snippets actually contain useful data or just meta descriptions
    # If total snippet text is mostly generic, fetch the best page
    import re as _re
    page_content = ""
    snippet_words = len(snippets.split())

    # Heuristic: check if snippets have actual specific data
    # Numbers with context (times, scores, prices) count. Generic "live scores available" doesn't.
    specific_patterns = _re.findall(r'\d{1,2}:\d{2}\s*(?:p\.m\.|a\.m\.|ET|PT)|\$[\d,.]+|\d+-\d+(?:\s*(?:win|loss|final))', snippets.lower())
    has_specifics = len(specific_patterns) >= 2  # need at least 2 specific data points

    if not has_specifics and all_results:
        # Snippets are weak — use Jina Reader to fetch the best page
        # Jina reads JS-rendered pages (ESPN, etc.) that urllib can't
        for r in all_results[:3]:
            url = r.get("href") or r.get("link", "")
            if not url:
                continue
            try:
                jina_url = f"https://r.jina.ai/{url}"
                req = urllib.request.Request(jina_url, headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/plain",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    text = resp.read(6000).decode("utf-8", errors="ignore")
                    if len(text) > 200:
                        page_content = text[:4000]
                        break
            except Exception:
                continue

    context = snippets
    if page_content:
        context += f"\n\nDetailed content from top result:\n{page_content}"

    # Step 3: LLM answers using results (~2-3s)
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. Answer the user's question using the search results below. Be specific, direct, and detailed. Extract dates, times, scores, names, numbers, prices, and facts. Present them clearly."},
        {"role": "user", "content": f"Search results:\n\n{context}\n\nQuestion: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0)

# ── code agent: tag-based tool calling ─────────────
# The model emits XML tags instead of a shell command, so file content travels
# as data rather than being smuggled through `sp.run(shell=True)`. Generation
# halts on the closing tag (llama-server `stop`), we run the tool, feed the
# result back, and repeat. Pattern borrowed from
# research/expert-sniper/distributed/mac_tensor/agent.py.
#
# Edits use verified SEARCH/REPLACE blocks: the SEARCH text must appear exactly
# once in the file, so a mis-remembered or paraphrased snippet is rejected
# instead of corrupting the file. Every write shows a diff and waits for y/n.

READ_LINE_CAP = 400
READ_CHAR_CAP = 12000
MAX_TOOL_ITERATIONS = 6

CODE_SYSTEM_PROMPT = """You are a coding agent working in {work_dir}. You read and change real files.

Answer the user by either calling ONE tool and stopping, or replying with a short final answer and no tags.

RULES
1. Emit exactly ONE tool call, then STOP. The system runs it and shows you the result.
2. Never write a <result> tag yourself. The system inserts results.
3. Before editing a file you have not already read in this conversation, <read> it first.
4. When you have enough information, answer in 1-3 sentences with no tags.
5. Do not repeat a tool call that already succeeded.

TOOLS
<read>path</read>                 read a file and get back its exact text
<ls>path</ls>                     list a directory
<shell>command</shell>            run a read-only shell command
<search>query</search>            web search
<write path="p">body</write>      create a NEW file (refused if it already exists)
<edit path="p">blocks</edit>      change an EXISTING file

EDIT FORMAT
<edit path="the/file.py">
<<<<<<< SEARCH
exact text copied from the file, character for character
=======
the replacement text
>>>>>>> REPLACE
</edit>

You may put several SEARCH/REPLACE blocks in one <edit>. The SEARCH text must
appear EXACTLY ONCE in the file, otherwise the edit is rejected and nothing is
written. To change something that appears twice, write two blocks, each with
more surrounding lines to make it unique. Never reformat, reorder, or
"improve" code you were not asked to change. Match the file's existing style.

EXAMPLE
User: change the greeting in greet.py to Spanish
You: <read>greet.py</read>
[system: <result>def greet():
    return "Hello"</result>]
You: <edit path="greet.py">
<<<<<<< SEARCH
    return "Hello"
=======
    return "Hola"
>>>>>>> REPLACE
</edit>
[system: <result>Updated greet.py (2 lines).</result>]
You: Changed the greeting to Spanish.

Now the user asks:
"""

# ── code agent: tools ───────────────────────────────
DESTRUCTIVE = ["rm ", "rm -", "mv ", "dd ", "mkfs", "chmod", "chown", "sudo",
               "kill", "shutdown", "reboot", "diskutil", "curl ", "wget ",
               "defaults delete", "> /dev/"]

def _resolve(path, work_dir="."):
    p = os.path.expanduser((path or "").strip().strip('"').strip("'"))
    if not p:
        p = "."
    if not os.path.isabs(p):
        p = os.path.join(work_dir, p)
    return os.path.normpath(p)

def tool_read(arg, work_dir="."):
    path = _resolve(arg, work_dir)
    if os.path.isdir(path):
        return f"{path} is a directory. Use <ls>{path}</ls>."
    if not os.path.isfile(path):
        return f"No such file: {path}"
    try:
        with open(path, "r", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"
    lines = raw.splitlines()
    total = len(lines)
    truncated = False
    if total > READ_LINE_CAP:
        lines = lines[:READ_LINE_CAP]
        truncated = True
    body = "\n".join(lines)
    if len(body) > READ_CHAR_CAP:
        body = body[:READ_CHAR_CAP]
        truncated = True
    out = f"{path} ({total} lines)\n{body}"
    if truncated:
        out += (f"\n... TRUNCATED. You are seeing the first {len(lines)} lines. "
                "Only edit text you can actually see here.")
    return out

def tool_ls(arg, work_dir="."):
    path = _resolve(arg or ".", work_dir)
    if not os.path.isdir(path):
        return f"Not a directory: {path}"
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return f"Error: {e}"
    lines = []
    for name in entries[:120]:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"{name}/")
        else:
            try:
                lines.append(f"{name}  ({os.path.getsize(full):,}b)")
            except OSError:
                lines.append(name)
    out = f"{path}\n" + "\n".join(lines)
    if len(entries) > 120:
        out += f"\n... and {len(entries) - 120} more"
    return out

def tool_shell(arg, work_dir="."):
    cmd = arg.strip()
    low = cmd.lower()
    for d in DESTRUCTIVE:
        if d in low:
            return (f"Refused (blocked: {d.strip()}). To change files use "
                    "<read> then <edit>. Use a different command for this.")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=30, cwd=work_dir)
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
    parts = []
    if r.stdout.strip():
        parts.append(r.stdout.strip()[:3000])
    if r.stderr.strip():
        parts.append(f"[stderr] {r.stderr.strip()[:800]}")
    parts.append(f"[exit {r.returncode}]")
    return "\n".join(parts)

def tool_search_raw(query):
    """Search without extra LLM round-trips — the loop feeds raw hits back."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Search unavailable: pip install ddgs"
    try:
        hits = list(DDGS().text(query, max_results=6))
    except Exception as e:
        return f"Search failed: {e}"
    if not hits:
        return "No results."
    out = []
    for h in hits[:6]:
        title = h.get("title", "")
        body = (h.get("body") or h.get("snippet") or "")[:220]
        url = h.get("href") or h.get("url") or ""
        out.append(f"• {title}\n  {body}\n  {url}")
    return "\n".join(out)

# ── code agent: verified edits ──────────────────────
BLOCK_RE = re.compile(
    r"<{5,}\s*SEARCH[ \t]*\r?\n(.*?)\r?\n\s*={5,}[ \t]*\r?\n(.*?)\r?\n\s*>{5,}\s*REPLACE",
    re.DOTALL,
)

def _closest_lines(old, lines):
    """Point the model at what is actually in the file so it can retry."""
    first = next((l.strip() for l in old.splitlines() if l.strip()), "")
    if not first:
        return ""
    key = re.sub(r"\W+", "", first)[:24].lower()
    if not key:
        return ""
    for i, ln in enumerate(lines):
        if key in re.sub(r"\W+", "", ln)[:len(key) + 10].lower():
            lo, hi = max(0, i - 1), min(len(lines), i + 3)
            body = "\n".join(f"  {j+1}: {lines[j]}" for j in range(lo, hi))
            return f"Closest line in the file:\n{body}"
    return f"No line in the file resembles {first!r}. Re-read the file before editing."

def _match_lines_rstrip(content_lines, old_lines):
    """Line-wise match ignoring trailing whitespace only.

    Leading whitespace must match exactly — an earlier \s+-based version
    re-applied the matched whitespace and doubled indentation, which silently
    corrupts Python. Returns the start index, "ambiguous", or None.
    """
    n, m = len(content_lines), len(old_lines)
    if m == 0 or m > n:
        return None
    hits = []
    for i in range(n - m + 1):
        if all(c.rstrip() == o.rstrip()
               for c, o in zip(content_lines[i:i + m], old_lines)):
            hits.append(i)
    if len(hits) == 1:
        return hits[0]
    return "ambiguous" if hits else None

def _replace_once(content, old, new, path):
    """Apply one SEARCH/REPLACE. Exact match only, uniqueness enforced."""
    if not old.strip():
        return False, content, "Refused: the SEARCH block was empty."

    n = content.count(old)
    if n == 1:
        return True, content.replace(old, new, 1), None
    if n > 1:
        return False, content, (
            f"The SEARCH text appears {n} times in {path} — ambiguous, nothing was "
            "written. Include more surrounding lines so it matches exactly one spot.")

    # Retry line-by-line, tolerating trailing whitespace only. Uniqueness is
    # still required, and the drift is surfaced in the diff before writing.
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    lines = content.splitlines(keepends=True)
    idx = _match_lines_rstrip(lines, old_lines)

    if idx == "ambiguous":
        return False, content, (
            f"The SEARCH text matches more than one place in {path} once trailing "
            "whitespace is ignored — ambiguous, nothing was written. Add more "
            "surrounding lines.")
    if idx is None:
        return False, content, (
            f"The SEARCH text was not found in {path} — nothing was written. "
            "It must match exactly, including indentation.\n"
            + _closest_lines(old, content.splitlines()))

    end = idx + len(old_lines)
    nl = "\r\n" if "\r\n" in "".join(lines[idx:end]) else "\n"
    had_final_nl = lines[end - 1].endswith(("\n", "\r"))
    block = "".join(l + nl for l in new_lines)
    if new_lines and not had_final_nl:
        block = block[: -len(nl)]
    return True, "".join(lines[:idx]) + block + "".join(lines[end:]), \
        "SEARCH matched ignoring trailing whitespace"

def _render_diff(path, original, new_content, label, notes):
    diff = list(difflib.unified_diff(
        (original if original is not None else "").splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{os.path.basename(path)}" if original is not None else "/dev/null",
        tofile=f"b/{os.path.basename(path)}",
        n=3,
    ))
    if not diff:
        return False

    console.print()
    head = Text()
    head.append(f"  {label}  ", style="bold bright_yellow")
    head.append(path, style="bold white")
    console.print(head)
    for note in (notes or []):
        console.print(f"    [dim]note: {note}[/]")
    console.print()

    for line in "".join(diff).rstrip("\n").split("\n"):
        if line.startswith(("+++", "---")):
            console.print(f"  [dim]{line}[/]")
        elif line.startswith("@@"):
            console.print(f"  [cyan]{line}[/]")
        elif line.startswith("+"):
            console.print(f"  [green]{line}[/]")
        elif line.startswith("-"):
            console.print(f"  [red]{line}[/]")
        else:
            console.print(f"  [dim]{line}[/]")
    console.print()
    return True

def _confirm_apply(path, new_content, original):
    """Ask on the main thread — callers must not be inside a worker thread."""
    prompt = Text()
    prompt.append("  apply this change?  ", style="bold bright_green")
    prompt.append("[y/N] ", style="dim")
    console.print(prompt, end="")
    try:
        ans = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    console.print()
    if ans not in ("y", "yes"):
        return f"Declined by the user — {path} was NOT modified."
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(new_content)
    except Exception as e:
        return f"Write failed: {e}"
    verb = "Created" if original is None else "Updated"
    return (f"{verb} {path} ({len(new_content.splitlines())} lines). "
            "Check it with <shell>git diff "
            f"{os.path.basename(path)}</shell> or read it back.")

def _commit(path, new_content, original, label, notes=None):
    if original is not None and new_content == original:
        return f"No change — {path} already matches. Nothing written."
    if not _render_diff(path, original, new_content, label, notes):
        return f"No change — {path} already matches. Nothing written."
    return _confirm_apply(path, new_content, original)

def tool_write(path_arg, content, work_dir="."):
    path = _resolve(path_arg, work_dir)
    if os.path.exists(path):
        return (f"Refused: {path} already exists. Use "
                f'<edit path="{path}"> to change it, or pick another name.')
    return _commit(path, content, None, f"create {os.path.basename(path)}")

def tool_edit(path_arg, body, work_dir="."):
    path = _resolve(path_arg, work_dir)
    if not os.path.isfile(path):
        return f"No such file: {path}. Use <write> to create it."
    try:
        with open(path, "r", errors="replace") as f:
            original = f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

    blocks = BLOCK_RE.findall(body)
    if not blocks:
        return ("No SEARCH/REPLACE block found. The required form is:\n"
                "<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE")

    content = original
    notes = []
    for old, new in blocks:
        ok, content, note = _replace_once(content, old, new, path)
        if not ok:
            return f"Edit rejected, nothing written.\n{note}"
        if note:
            notes.append(note)

    return _commit(path, content, original, f"edit {os.path.basename(path)}", notes)

# ── code agent: parser + loop ───────────────────────
TOOL_REGEX = re.compile(r"<(read|ls|search|shell)>(.+?)</\1>", re.DOTALL)
ATTR_REGEX = re.compile(r'<(write|edit)\s+path="([^"]+)"\s*>(.*?)</\1>', re.DOTALL)
CODE_STOP = ["</read>", "</ls>", "</search>", "</shell>", "</write>", "</edit>"]

def parse_code_tool(text):
    """Find the first complete tool call in the model's output."""
    m = ATTR_REGEX.search(text)
    if m:
        return {"tool": m.group(1), "path": m.group(2), "body": m.group(3),
                "start": m.start(), "end": m.end()}
    m = TOOL_REGEX.search(text)
    if m:
        return {"tool": m.group(1), "content": m.group(2),
                "start": m.start(), "end": m.end()}
    return None

def run_code_tool(call, work_dir="."):
    t = call["tool"]
    if t == "read":   return tool_read(call["content"], work_dir)
    if t == "ls":     return tool_ls(call["content"], work_dir)
    if t == "shell":  return tool_shell(call["content"], work_dir)
    if t == "search": return tool_search_raw(call["content"])
    if t == "write":  return tool_write(call["path"], call["body"], work_dir)
    if t == "edit":   return tool_edit(call["path"], call["body"], work_dir)
    return f"Unknown tool: {t}"

def llm_call_code(convo, max_tokens=3000, temperature=0.2):
    payload = json.dumps({
        "model": "local",
        "messages": convo,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": CODE_STOP,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return d["choices"][0]["message"]["content"], d.get("timings", {})

def run_code_agent(user_input, work_dir=".", history=None,
                   max_iters=MAX_TOOL_ITERATIONS):
    """Tag-based tool loop.

    Returns (final_text, tokens, elapsed, tools_used, convo). `convo` carries the
    full exchange including tool results so the caller can persist it as history.
    """
    convo = [{"role": "system", "content": CODE_SYSTEM_PROMPT.format(work_dir=work_dir)}]
    # Drop any system message from history (e.g. one set via /system) — two
    # system prompts in one request makes the smaller models behave erratically.
    convo += [dict(m) for m in (history or []) if m.get("role") != "system"]
    convo.append({"role": "user", "content": user_input})

    used = []
    tokens = 0
    start = time.time()
    final = ""
    limit_hit = False

    for step in range(max_iters):
        out, err = [None], [None]

        def work():
            try:
                out[0] = llm_call_code(convo)
            except Exception as e:
                err[0] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()

        display = WorkingDisplay()
        display.phase = "thinking" if step == 0 else "deciding next step"
        with Live(display.render(), console=console,
                  refresh_per_second=8, transient=True) as live:
            while th.is_alive():
                display.frame += 1
                live.update(display.render())
                time.sleep(0.12)
        th.join(timeout=1)

        if err[0] is not None or out[0] is None:
            final = f"[error] {err[0] or 'no response from model'}"
            break

        text, timings = out[0]
        tokens += timings.get("predicted_n") or 0
        call = parse_code_tool(text)

        if not call:
            final = text.strip()
            break

        preamble = text[:call["start"]].strip()
        if preamble:
            console.print(f"  [dim]{preamble[:200]}[/]")

        target = call.get("path") or (call.get("content") or "").strip()[:40]
        console.print(f"  [dim]▸ {call['tool']} {os.path.basename(target)}[/]")

        result = run_code_tool(call, work_dir)
        used.append(call["tool"])

        convo.append({"role": "assistant", "content": text.strip()})
        convo.append({"role": "user", "content": f"<result>\n{result}\n</result>"})

        if step == max_iters - 1:
            limit_hit = True

    if limit_hit and not final:
        final = ("Stopped at the tool-call limit. The last tool ran — ask me to "
                 "continue if the job is not finished.")

    return final, tokens, time.time() - start, used, convo

def get_current_model():
    """Check which model the running server has loaded."""
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "35b"
        elif "9B" in alias:
            return "9b"
    except Exception:
        pass
    return None

def swap_model(target_key):
    """Stop current server and start a new one with the target model."""
    cfg = MODELS[target_key]
    if not os.path.exists(cfg["path"]):
        return False, f"Model not found: {cfg['path']}"

    # Kill current server
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

    # Start new server
    cmd_list = [
        "llama-server",
        "--model", cfg["path"],
        "--port", "8000",
        "--host", "127.0.0.1",
        "--ctx-size", str(cfg["ctx"]),
    ] + cfg["flags"].split()
    subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for ready
    for i in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{SERVER}/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                return True, f"Switched to {cfg['name']} ({cfg['ctx']} ctx)"
        except Exception:
            pass

    return False, "Server failed to start"

# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')
def strip_ansi(text):
    return ANSI_RE.sub('', text)

# ── live working display ──────────────────────────
DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

class WorkingDisplay:
    def __init__(self):
        self.events = []
        self.phase = "thinking"
        self.frame = 0
        self.start_time = time.time()
        self.logs = []

    def add_log(self, line):
        clean = strip_ansi(line).strip()
        if not clean:
            return

        lower = clean.lower()
        new_phase = None
        detail = ""

        if "processing message" in lower:
            new_phase = "reading your message"
        elif "llm_request" in lower:
            new_phase = "thinking"
        elif "tool_call" in lower or "web_search" in lower:
            if "web_search" in lower or "duckduckgo" in lower:
                new_phase = "searching the web"
            elif "web_fetch" in lower or "fetch" in lower:
                new_phase = "fetching page"
            elif "exec" in lower:
                new_phase = "running command"
            elif "read_file" in lower:
                new_phase = "reading file"
            elif "write_file" in lower:
                new_phase = "writing file"
            else:
                new_phase = "using tools"
        elif "context_compress" in lower:
            new_phase = "compressing context"
        elif "turn_end" in lower:
            new_phase = "finishing up"

        if new_phase:
            self.phase = new_phase
            self.events.append((time.time() - self.start_time, new_phase, detail))

        # Keep last few interesting log lines
        if any(k in lower for k in ["llm_request", "tool_call", "tool_result", "turn_end", "web_search", "fetch", "exec"]):
            short = clean
            if ">" in short:
                short = short.split(">", 1)[-1].strip()
            if len(short) > 70:
                short = short[:67] + "..."
            self.logs.append(short)
            if len(self.logs) > 3:
                self.logs.pop(0)

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start_time
        spinner = DOTS[self.frame % len(DOTS)]

        t = Text()
        t.append(f"  {spinner} ", style="bold bright_cyan")
        t.append(self.phase, style="bold bright_cyan")
        t.append(f"  {elapsed:.0f}s", style="dim")
        t.append("\n")

        for log in self.logs[-3:]:
            t.append(f"    {log}\n", style="dim italic")

        return t

# ── detect model ───────────────────────────────────
def detect_model():
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "Qwen3.5-35B-A3B", "MoE 34.7B · 3B active · IQ2_M"
        elif "9B" in alias:
            return "Qwen3.5-9B", "8.95B dense · Q4_K_M"
        return alias.replace(".gguf", "").split("/")[-1], "local"
    except Exception:
        return "offline", ""

# ── streaming chat (raw mode) ─────────────────────
def stream_llm(messages):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        while True:
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return full, tokens, time.time() - start
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c
                except Exception:
                    pass

    return full, tokens, time.time() - start

# ── picoclaw agent call with LIVE log streaming ───
def picoclaw_call_live(message, session="mac-code"):
    """Run picoclaw with real-time log streaming into animated display."""
    cmd = [PICOCLAW, "agent", "-m", message, "-s", session]
    display = WorkingDisplay()
    all_lines = []

    # Launch with Popen — picoclaw writes everything to stdout
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    # Read stdout line-by-line in a thread for real-time updates
    def read_output():
        try:
            for line in proc.stdout:
                all_lines.append(line)
                display.add_log(line)
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    # Animate while process runs
    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
        while proc.poll() is None:
            live.update(display.render())
            time.sleep(0.12)
        # Give reader a moment to finish
        time.sleep(0.3)
        live.update(display.render())

    reader.join(timeout=2)

    # Parse: strip ANSI, find lobster emoji, take text after it
    raw = "".join(all_lines)
    clean = strip_ansi(raw)

    idx = clean.rfind("\U0001f99e")  # last lobster emoji
    if idx >= 0:
        response = clean[idx:].lstrip("\U0001f99e").strip()
        # If it starts with "Error:" it's a picoclaw error, not a model response
        if response.startswith("Error:"):
            # Extract the useful part of the error
            response = f"[agent error] {response[:200]}"
    else:
        # No lobster — take non-banner lines
        lines = clean.split("\n")
        resp = []
        past = False
        for line in lines:
            s = line.strip()
            if not past:
                if not s or any(c in s for c in ["██", "╔", "╚", "╝", "║"]):
                    continue
                past = True
            if past and s:
                resp.append(s)
        response = "\n".join(resp).strip()

    return response, display.events

# ── banner ─────────────────────────────────────────
def print_banner(model_name, model_detail):
    console.print()
    logo = Text()
    logo.append("  \U0001f34e ", style="default")
    logo.append("mac", style="bold bright_cyan")
    logo.append(" ", style="default")
    logo.append("code", style="bold bright_yellow")
    console.print(logo)

    sub = Text()
    sub.append("  claude code, but it runs on your Mac for free", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", model_name, model_detail),
        ("tools", "search · fetch · exec · files", ""),
        ("cost", "$0.00/hr", "Apple M4 Metal · localhost:8000"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:6s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print("  [dim]type [bold bright_cyan]/[/bold bright_cyan] to see all commands[/]\n")

# ── render helpers ─────────────────────────────────
def render_response(response):
    """Render a response — use Rich Markdown if it has formatting, plain text otherwise."""
    if any(c in response for c in ["##", "**", "```", "| ", "- ", "1. ", "* "]):
        console.print(Padding(Markdown(response), (0, 2)))
    else:
        for line in response.split("\n"):
            console.print(f"  {line}")

def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s = Text()
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)

def render_timeline(events):
    """Show a compact summary of what the agent did."""
    if not events:
        return
    summary = []
    last_phase = None
    for ts, phase, detail in events:
        if phase != last_phase:
            summary.append(phase)
            last_phase = phase

    if len(summary) <= 1:
        return

    t = Text()
    t.append("  ", style="dim")
    for i, phase in enumerate(summary):
        t.append(phase, style="dim italic")
        if i < len(summary) - 1:
            t.append(" → ", style="dim")
    console.print(t)

# ── commands ───────────────────────────────────────
COMMANDS = [
    ("/agent",       "Switch to agent mode (tools + web search)"),
    ("/raw",         "Switch to raw mode (direct streaming, no tools)"),
    ("/btw",         "Ask a side question without adding to conversation history"),
    ("/loop",        "Run a prompt on a recurring interval — /loop 5m <prompt>"),
    ("/branch",      "Save conversation checkpoint you can restore later"),
    ("/restore",     "Restore last saved conversation checkpoint"),
    ("/add-dir",     "Set working directory — /add-dir <path>"),
    ("/save",        "Save conversation to a file — /save <filename>"),
    ("/search",      "Quick web search — /search <query>"),
    ("/bench",       "Run a quick speed benchmark"),
    ("/clear",       "Clear conversation and start fresh"),
    ("/stats",       "Show session statistics"),
    ("/model",       "Show or switch model — /model 9b or /model 35b"),
    ("/auto",        "Toggle smart auto-routing between 9B and 35B"),
    ("/tools",       "List available agent tools"),
    ("/system",      "Set system prompt — /system <message>"),
    ("/compact",     "Toggle compact output (no markdown rendering)"),
    ("/stop",        "Stop a running /loop"),
    ("/cost",        "Show estimated cost savings vs cloud APIs"),
    ("/good",        "Grade last response as good (for self-improvement)"),
    ("/bad",         "Grade last response as bad (for self-improvement)"),
    ("/improve",     "Show self-improvement stats from logged interactions"),
    ("/quit",        "Exit mac code"),
]

def show_slash_menu(filter_text=""):
    """Print slash commands inline — like Claude Code."""
    matches = COMMANDS
    if filter_text and filter_text != "/":
        matches = [(c, d) for c, d in COMMANDS if c.startswith(filter_text)]

    for cmd, desc in matches:
        line = Text()
        line.append(f"  {cmd}", style="bold bright_cyan")
        pad = " " * max(14 - len(cmd), 1)
        line.append(pad)
        line.append(desc, style="dim")
        console.print(line)

# ── main ───────────────────────────────────────────
def main():
    model_name, model_detail = detect_model()
    console.clear()
    print_banner(model_name, model_detail)

    messages = []
    session_tokens = 0
    session_time = 0.0
    session_turns = 0
    session_id = f"mc-{int(time.time())}"
    use_agent = True
    compact_mode = False
    auto_route = True  # smart routing between 9B and 35B
    work_dir = os.getcwd()
    branch_save = None
    loop_thread = None
    loop_running = False
    last_interaction = None  # for /good /bad grading

    while True:
        try:
            cur = get_current_model() or "?"
            tag = f"{'auto' if auto_route else 'agent'} {cur}" if use_agent else "raw"
            console.print(f"  [dim]{tag}[/] [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip()
        cmd_lower = cmd.lower()

        # ── slash command handling ─────────────
        if cmd == "/":
            show_slash_menu()
            continue
        elif cmd_lower.startswith("/") and not cmd_lower.startswith("/system "):
            # Check for partial match — typing "/st" shows "/stats" and "/system"
            exact = cmd_lower.split()[0]

            if exact in ("/quit", "/exit", "/q"):
                break
            elif exact == "/clear":
                messages.clear()
                session_id = f"mc-{int(time.time())}"
                console.clear()
                print_banner(model_name, model_detail)
                console.print("  [dim]cleared.[/]\n")
                continue
            elif exact == "/stats":
                avg = session_tokens / session_time if session_time > 0 else 0
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=12)
                t.add_column()
                t.add_row("turns", str(session_turns))
                t.add_row("tokens", f"{session_tokens:,}")
                t.add_row("time", f"{session_time:.1f}s")
                t.add_row("avg speed", f"{avg:.1f} tok/s")
                t.add_row("mode", tag)
                console.print(t)
                console.print()
                continue
            elif exact == "/model":
                # Check if user passed an argument like "/model 9b"
                parts = cmd.split()
                if len(parts) >= 2:
                    target = parts[1].lower().replace("b", "b")
                    if target in MODELS:
                        console.print(f"  [dim]swapping to {MODELS[target]['name']}...[/]")
                        display = WorkingDisplay()
                        display.phase = f"loading {MODELS[target]['name']}"
                        with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                            ok, msg = swap_model(target)
                            while not ok and display.frame < 100:
                                display.frame += 1
                                live.update(display.render())
                                time.sleep(0.2)
                        if ok:
                            model_name = MODELS[target]["name"]
                            model_detail = MODELS[target]["detail"]
                            console.print(f"  [bold bright_green]{msg}[/]\n")
                        else:
                            console.print(f"  [bold red]{msg}[/]\n")
                    else:
                        console.print(f"  [dim]available: 9b, 35b[/]\n")
                else:
                    cur = get_current_model()
                    model_name, model_detail = detect_model()
                    console.print(f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]")
                    console.print(f"  [dim]auto-routing: {'on' if auto_route else 'off'}[/]")
                    console.print(f"  [dim]switch: /model 9b  or  /model 35b[/]\n")
                continue

            elif exact == "/auto":
                auto_route = not auto_route
                state = "on" if auto_route else "off"
                console.print(f"  [dim]smart auto-routing {state}[/]")
                if auto_route:
                    console.print(f"  [dim]  tools/search → 9B (32K ctx, reliable)[/]")
                    console.print(f"  [dim]  reasoning     → 35B (faster, smarter)[/]")
                console.print()
                continue
            elif exact == "/tools":
                for name, desc in [
                    ("web_search", "DuckDuckGo"), ("web_fetch", "read URLs"),
                    ("exec", "shell commands"), ("read_file", "local files"),
                    ("write_file", "create files"), ("edit_file", "modify files"),
                    ("list_dir", "browse dirs"), ("subagent", "spawn tasks"),
                ]:
                    t = Text()
                    t.append("  ▸ ", style="bright_cyan")
                    t.append(name, style="bold bright_cyan")
                    t.append(f"  {desc}", style="dim")
                    console.print(t)
                console.print("  [dim]code mode (<read> <edit> <write> <ls> <shell> <search>) "
                              "handles file changes[/]\n")
                continue
            elif exact == "/agent":
                use_agent = True
                console.print("  [dim]agent mode (tools enabled)[/]\n")
                continue
            elif exact == "/raw":
                use_agent = False
                console.print("  [dim]raw mode (streaming, no tools)[/]\n")
                continue
            elif exact == "/compact":
                compact_mode = not compact_mode
                state = "on" if compact_mode else "off"
                console.print(f"  [dim]compact mode {state}[/]\n")
                continue

            elif exact == "/branch":
                branch_save = [m.copy() for m in messages]
                console.print(f"  [dim]conversation saved ({len(messages)} messages). use /restore to go back.[/]\n")
                continue

            elif exact == "/restore":
                if branch_save is not None:
                    messages = [m.copy() for m in branch_save]
                    console.print(f"  [dim]restored to checkpoint ({len(messages)} messages)[/]\n")
                else:
                    console.print("  [dim]no checkpoint saved. use /branch first.[/]\n")
                continue

            elif exact == "/bench":
                console.print("  [dim]running speed benchmark...[/]")
                try:
                    payload = json.dumps({
                        "model": "local",
                        "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
                        "max_tokens": 300, "temperature": 0.1,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    bstart = time.time()
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        d = json.loads(resp.read())
                    belapsed = time.time() - bstart
                    t = d.get("timings", {})
                    u = d.get("usage", {})
                    gen_speed = t.get("predicted_per_second", 0)
                    prompt_speed = t.get("prompt_per_second", 0)
                    tokens = u.get("completion_tokens", 0)
                    console.print(f"  [bold bright_green]{gen_speed:.1f} tok/s[/] generation")
                    console.print(f"  [bold bright_green]{prompt_speed:.1f} tok/s[/] prompt processing")
                    console.print(f"  [dim]{tokens} tokens in {belapsed:.1f}s[/]\n")
                except Exception as e:
                    console.print(f"  [bold red]benchmark failed: {e}[/]\n")
                continue

            elif exact == "/cost":
                cloud_rate = 0.34  # $/hr RunPod equivalent
                hours = session_time / 3600 if session_time > 0 else 0
                saved = cloud_rate * max(hours, 1/60)
                console.print(f"  [bold bright_green]$0.00[/] spent locally")
                console.print(f"  [dim]~${saved:.4f} would have cost on cloud GPU (${cloud_rate}/hr)[/]")
                console.print(f"  [dim]session: {session_time:.0f}s · {session_tokens:,} tokens[/]\n")
                continue

            elif exact == "/good":
                # Grade last response as good
                if last_interaction:
                    last_interaction["grade"] = "good"
                    log_interaction(**last_interaction)
                    console.print("  [bright_green]marked good[/]\n")
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/bad":
                # Grade last response as bad
                if last_interaction:
                    last_interaction["grade"] = "bad"
                    log_interaction(**last_interaction)
                    console.print("  [bright_red]marked bad — logged for improvement[/]\n")
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/improve":
                stats = get_failure_stats()
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=14)
                t.add_column()
                t.add_row("total", str(stats["total"]))
                t.add_row("good", str(stats["graded"].get("good", 0)))
                t.add_row("bad", str(stats["graded"].get("bad", 0)))
                t.add_row("errors", str(stats["errors"]))
                t.add_row("searches", str(stats["intents"].get("search", 0)))
                t.add_row("shell", str(stats["intents"].get("shell", 0)))
                t.add_row("code", str(stats["intents"].get("code", 0)))
                t.add_row("chat", str(stats["intents"].get("chat", 0)))
                t.add_row("logs", str(LOGS_DIR))
                console.print(t)
                console.print()
                continue

            elif exact in ("/help", "/?"):
                show_slash_menu()
                continue
            else:
                # Partial match — show filtered results
                show_slash_menu(exact)
                continue

        # ── commands with arguments ────────────
        elif cmd_lower.startswith("/system "):
            sys_msg = cmd[8:].strip()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]system: {sys_msg[:80]}[/]\n")
            continue

        elif cmd_lower.startswith("/btw "):
            # Side question — don't add to conversation history
            side_q = cmd[5:].strip()
            if not side_q:
                console.print("  [dim]/btw <question>[/]\n")
                continue
            console.print()
            if use_agent:
                start = time.time()
                # Use a separate session so it doesn't pollute main conversation
                response, events = picoclaw_call_live(side_q, session=f"btw-{int(time.time())}")
                elapsed = time.time() - start
                if response:
                    console.print(f"  [dim italic](side answer)[/]")
                    render_response(response)
                    console.print()
                    tokens_est = len(response.split())
                    render_speed(tokens_est, elapsed)
                    session_tokens += tokens_est
                    session_time += elapsed
            else:
                side_msgs = [{"role": "user", "content": side_q}]
                try:
                    payload = json.dumps({
                        "model": "local", "messages": side_msgs,
                        "max_tokens": 2000, "temperature": 0.7,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        d = json.loads(resp.read())
                    content = d["choices"][0]["message"]["content"]
                    console.print(f"  [dim italic](side answer)[/]")
                    for line in content.split("\n"):
                        console.print(f"  {line}")
                except Exception as e:
                    console.print(f"  [bold red]{e}[/]")
            console.print()
            continue

        elif cmd_lower.startswith("/add-dir "):
            new_dir = os.path.expanduser(cmd[9:].strip())
            if os.path.isdir(new_dir):
                work_dir = new_dir
                os.chdir(work_dir)
                console.print(f"  [dim]working directory: {work_dir}[/]\n")
            else:
                console.print(f"  [bold red]not a directory: {new_dir}[/]\n")
            continue

        elif cmd_lower.startswith("/save "):
            filename = cmd[6:].strip()
            if not filename:
                filename = f"conversation-{int(time.time())}.json"
            try:
                save_path = os.path.join(work_dir, filename)
                with open(save_path, "w") as f:
                    json.dump({
                        "messages": messages,
                        "session_id": session_id,
                        "tokens": session_tokens,
                        "time": session_time,
                        "turns": session_turns,
                    }, f, indent=2)
                console.print(f"  [dim]saved to {save_path}[/]\n")
            except Exception as e:
                console.print(f"  [bold red]{e}[/]\n")
            continue

        elif cmd_lower.startswith("/search "):
            query = cmd[8:].strip()
            if not query:
                console.print("  [dim]/search <query>[/]\n")
                continue
            console.print()
            start = time.time()
            response, events = picoclaw_call_live(
                f"Search the web for: {query}. Give a brief summary of the top results.",
                session=f"search-{int(time.time())}"
            )
            elapsed = time.time() - start
            if response:
                for line in response.split("\n"):
                    console.print(f"  {line}")
                console.print()
                s = Text()
                s.append(f"  ▸ agent", style="bold bright_cyan")
                s.append(f"  {elapsed:.1f}s total (search + inference)", style="dim")
                console.print(s)
                session_tokens += len(response.split())
                session_time += elapsed
            console.print()
            continue

        elif cmd_lower.startswith("/loop "):
            # Parse: /loop 5m <prompt>
            parts = cmd[6:].strip().split(None, 1)
            if len(parts) < 2:
                console.print("  [dim]/loop <interval> <prompt>  — e.g. /loop 5m check server status[/]\n")
                continue

            interval_str, loop_prompt = parts
            # Parse interval
            try:
                if interval_str.endswith("m"):
                    interval_sec = int(interval_str[:-1]) * 60
                elif interval_str.endswith("s"):
                    interval_sec = int(interval_str[:-1])
                elif interval_str.endswith("h"):
                    interval_sec = int(interval_str[:-1]) * 3600
                else:
                    interval_sec = int(interval_str) * 60  # default minutes
            except ValueError:
                console.print(f"  [bold red]invalid interval: {interval_str}[/]\n")
                continue

            if loop_running:
                loop_running = False
                console.print("  [dim]stopped previous loop[/]")
                time.sleep(1)

            loop_running = True
            console.print(f"  [dim]looping every {interval_sec}s: {loop_prompt}[/]")
            console.print(f"  [dim]type /stop to cancel[/]\n")

            def run_loop(prompt, interval, sid):
                nonlocal loop_running, session_tokens, session_time
                while loop_running:
                    time.sleep(interval)
                    if not loop_running:
                        break
                    console.print(f"\n  [dim italic]loop: running '{prompt[:40]}...'[/]")
                    resp, _ = picoclaw_call_live(prompt, session=sid)
                    if resp:
                        for line in resp.split("\n"):
                            console.print(f"  {line}")
                    console.print()

            loop_thread = threading.Thread(
                target=run_loop,
                args=(loop_prompt, interval_sec, f"loop-{session_id}"),
                daemon=True
            )
            loop_thread.start()
            continue

        elif cmd_lower == "/stop":
            if loop_running:
                loop_running = False
                console.print("  [dim]loop stopped[/]\n")
            else:
                console.print("  [dim]no loop running[/]\n")
            continue

        console.print()

        # ── agent mode ─────────────────────────────
        if use_agent:
            start = time.time()

            # LLM classifies intent: search, shell, or chat (~1s)
            display = WorkingDisplay()
            display.phase = "classifying"
            intent_result = [None]

            def do_classify():
                intent_result[0] = classify_intent(user_input)

            cls_thread = threading.Thread(target=do_classify, daemon=True)
            cls_thread.start()

            with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                while cls_thread.is_alive():
                    display.frame += 1
                    live.update(display.render())
                    time.sleep(0.12)

            cls_thread.join(timeout=1)
            intent = intent_result[0] or "chat"

            # Safety net: a mis-routed edit request is invisible — the model
            # just answers in chat and no file is ever touched. Catch the
            # unambiguous ones and send them to the code agent instead.
            if intent in ("chat", "shell") and looks_like_edit_request(user_input):
                intent = "code"

            # Route based on LLM classification
            if intent == "shell":
                # File/system operations → LLM generates shell command
                display = WorkingDisplay()
                display.phase = "running command"
                tool_result = [None]

                def do_tool():
                    try:
                        tool_result[0] = run_smart_tool(user_input, work_dir)
                    except Exception as e:
                        tool_result[0] = None

                tool_thread = threading.Thread(target=do_tool, daemon=True)
                tool_thread.start()

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    while tool_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "generating command"
                        elif t < 5:
                            display.phase = "executing"
                        else:
                            display.phase = "reading results"
                        live.update(display.render())
                        time.sleep(0.12)

                tool_thread.join(timeout=1)
                result = tool_result[0]
                elapsed = time.time() - start

                if result:
                    response, speed, cmd = result
                    console.print()
                    console.print(f"  [dim]$ {cmd}[/]")
                    console.print()
                    render_response(response)
                    console.print()
                    s = Text()
                    s.append(f"  ▸ shell", style="bold bright_cyan")
                    s.append(f"  {elapsed:.1f}s", style="dim")
                    if speed > 0:
                        s.append(f"  ·  {speed:.1f} tok/s", style="bright_green")
                    console.print(s)
                    session_tokens += len(response.split())
                    session_time += elapsed
                    session_turns += 1
                    last_interaction = {"query": user_input, "intent": "shell", "response": response, "speed": speed}
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                else:
                    console.print(f"  [bold red]command failed[/]\n")

            elif intent == "search":
                # Web search → fast direct path (~3-5s)
                display = WorkingDisplay()
                display.phase = "searching the web"
                search_result = [None]

                def do_search():
                    try:
                        search_result[0] = quick_search(user_input)
                    except Exception:
                        search_result[0] = None

                search_thread = threading.Thread(target=do_search, daemon=True)
                search_thread.start()

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    while search_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "rewriting query"
                        elif t < 3:
                            display.phase = "searching the web"
                        else:
                            display.phase = "generating answer"
                        live.update(display.render())
                        time.sleep(0.12)

                search_thread.join(timeout=1)
                result = search_result[0]
                elapsed = time.time() - start

                if result:
                    response, speed = result
                    console.print()
                    render_response(response)
                    console.print()
                    s = Text()
                    s.append(f"  ▸ search", style="bold bright_cyan")
                    s.append(f"  {elapsed:.1f}s", style="dim")
                    if speed > 0:
                        s.append(f"  ·  {speed:.1f} tok/s", style="bright_green")
                    console.print(s)
                    session_tokens += len(response.split())
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                    last_interaction = {"query": user_input, "intent": "search", "response": response, "speed": speed}
                else:
                    # Search failed, fall back to direct LLM
                    console.print("  [dim]search failed, asking model directly...[/]")
                    messages.append({"role": "user", "content": user_input})
                    full = ""
                    tokens = 0
                    first_token = True
                    display.phase = "thinking"
                    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                        gen = stream_llm(with_system(messages, work_dir))
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})

            elif intent == "code":
                # File create/edit → tag-based tool loop, chains read → edit
                try:
                    response, tokens, elapsed, used, convo = run_code_agent(
                        user_input, work_dir, history=messages
                    )
                except Exception as e:
                    console.print(f"  [bold red]{e}[/]")
                    continue

                console.print()
                if response:
                    render_response(response)
                console.print()

                speed = tokens / elapsed if elapsed > 0 else 0
                s = Text()
                s.append("  ▸ code", style="bold bright_cyan")
                s.append(f"  {elapsed:.1f}s", style="dim")
                if speed > 0:
                    s.append(f"  {speed:.1f} tok/s", style="bright_green")
                if used:
                    s.append(f"  ·  {len(used)} tool call"
                             f"{'s' if len(used) != 1 else ''}", style="dim")
                    s.append(f" ({', '.join(used)})", style="dim italic")
                console.print(s)

                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                last_interaction = {"query": user_input, "intent": "code",
                                    "response": response, "speed": speed}
                # Persist the tool traffic so follow-ups keep the file context.
                messages[:] = [m for m in convo if m["role"] != "system"]

            else:
                # Direct LLM streaming (no tools needed)
                messages.append({"role": "user", "content": user_input})
                full = ""
                tokens = 0
                first_token = True
                display = WorkingDisplay()
                display.phase = "thinking"
                try:
                    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                        gen = stream_llm(with_system(messages, work_dir))
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    console.print(f"\n  [bold red]{e}[/]")
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()

        # ── raw streaming mode ─────────────────────
        else:
            messages.append({"role": "user", "content": user_input})
            full = ""
            tokens = 0
            start = time.time()

            try:
                display = WorkingDisplay()
                display.phase = "thinking"
                first_token = True

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    gen = stream_llm(with_system(messages, work_dir))
                    for chunk in gen:
                        if isinstance(chunk, str):
                            if first_token:
                                first_token = False
                                live.stop()
                                console.print("  ", end="")
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1

                elapsed = time.time() - start
                if not compact_mode and any(c in full for c in ["##", "**", "```", "- ", "1. "]):
                    console.print("\n")
                    console.print(Padding(Markdown(full), (0, 2)))
                else:
                    console.print("\n")
                render_speed(tokens, elapsed)
                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "assistant", "content": full})

            except Exception as e:
                console.print(f"  [bold red]{e}[/]")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

        console.print()

    # ── exit ───────────────────────────────────────
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  \U0001f34e [bold bright_cyan]mac[/] [bold bright_yellow]code[/]"
            f"  [dim]{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} tok/s[/]"
        )
    console.print()

if __name__ == "__main__":
    main()
