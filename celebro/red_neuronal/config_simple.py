"""
Configuración Simple de @red_neuronal
Versión: 0.6.0
"""

# API Keys
GEMINI_API_KEY = "# aqui tu  api de  gemini #"

# Configuración de Gemini API
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_MAX_TOKENS = 2048
GEMINI_TEMPERATURE = 0.7
TIMEOUT = 30
RETRY_ATTEMPTS = 3

def get_gemini_api_key():
    """Obtiene la API key de Gemini"""
    return GEMINI_API_KEY

def get_gemini_config():
    """Obtiene la configuración de Gemini API"""
    return {
        "api_key": GEMINI_API_KEY,
        "model": GEMINI_MODEL,
        "max_tokens": GEMINI_MAX_TOKENS,
        "temperature": GEMINI_TEMPERATURE,
        "timeout": TIMEOUT,
        "retry_attempts": RETRY_ATTEMPTS
    }
