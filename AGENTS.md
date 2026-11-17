# Course LLM logging rules (required)
For EVERY interaction related to this repository, including brainstorming, setup, coding,
debugging, analysis, and report writing:
1. Preserve the COMPLETE user prompt and COMPLETE assistant/agent response verbatim,
   even when output is discarded, including any tool session transcript when available.
2. In `llm_log/`, store date/time (with timezone), team member, tool, model,
   full prompt, full response, and files created/modified with line ranges.
3. Append entries; never overwrite or delete previous logs.
4. Commit log updates together with associated source modifications.
5. If the tool cannot capture its own full response, explicitly instruct the team member
   to export/copy the full interaction immediately. Do NOT claim automatic logging.
6. Never log passwords, SSH private keys, authentication tokens, or sensitive secrets.
7. All team members must understand and verify AI-generated code before committing it.
