"""Story -> video: a written story becomes scenes, generated clips, a timed Hindi
script and, through the existing SRT dub path, a finished dubbed video.

    script.py     the story file format (cast, scenes, shots, lines) + SRT export
    breakdown.py  plain text -> Story, with an LLM (Groq/Gemini) in one JSON pass
    generate.py   clips for every shot, Wan 2.1 (local GPU) in a child process
    assemble.py   clips -> one video, lines -> SRT with [SPEAKER] labels
"""
