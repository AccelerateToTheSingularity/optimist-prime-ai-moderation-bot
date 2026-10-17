# Dependencies and external services

The source supports Python 3.11 or newer. Install its Python dependencies from the repository root:

```powershell
py -m pip install -r requirements.txt
```

| Package | Purpose |
| --- | --- |
| PRAW | Reddit API access |
| `python-dotenv` | Loading local environment settings in standalone tools |
| `cryptography` | Optional encrypted local state |
| PySide6 | Local moderator console and settings window |
| PyYAML | Supported local YAML configuration |

## AI services

The structured-decision client uses OpenRouter. Set `OPENROUTER_API_KEY` to use features that make structured-decision requests.

Text generation uses an OpenAI-compatible chat API. The source includes provider presets for MiniMax, OpenAI, Claude, Gemini, DeepSeek, GLM, Groq, Mistral, Together, xAI, and a custom compatible endpoint. `BOT_LLM_PROVIDER` selects a preset, `LLM_MODEL` selects a model, and the selected provider's API key must be set. For a custom endpoint, set `OPENAI_BASE_URL` or `LLM_BASE_URL` and the matching API key.

Provider model names, endpoint behavior, and terms can change. Check the selected provider's current documentation before setup. Optimist Prime uses a standard-library HTTP client for text generation and does not require the OpenAI Python SDK.

Reddit content may be sent to the selected AI service. Review the service's data-handling terms before enabling features that send content.

The repository does not vendor provider or dependency source. Each third-party package and service remains subject to its own terms and license.

The project source uses the MIT License, but PySide6/Qt for Python has separate LGPL, GPL, and commercial licensing options. This repository does not bundle Qt. Anyone packaging or redistributing an installed application should check the license and obligations for the exact Qt components they ship using the [official Qt for Python licensing guide](https://doc.qt.io/qtforpython-6/commercial/index.html).
