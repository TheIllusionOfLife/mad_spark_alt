## 2025-02-08 - [Insecure Environment Variable Parsing]
**Vulnerability:** The application was manually parsing the `.env` file, which led to incorrect handling of inline comments (including them in values) and potentially mishandling quoted values.
**Learning:** Manual parsing of configuration files is error-prone and can introduce security vulnerabilities or misconfigurations. Established libraries like `python-dotenv` handle edge cases (escaping, quotes, comments) robustly.
**Prevention:** Always use `python-dotenv`'s `load_dotenv` for loading environment variables from `.env` files. Ensure tests cover edge cases like special characters and comments in configuration values.
