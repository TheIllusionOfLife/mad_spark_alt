## 2025-02-08 - [Insecure Environment Variable Parsing]
**Vulnerability:** The application was manually parsing the `.env` file, which led to incorrect handling of inline comments (including them in values) and potentially mishandling quoted values.
**Learning:** Manual parsing of configuration files is error-prone and can introduce security vulnerabilities or misconfigurations. Established libraries like `python-dotenv` handle edge cases (escaping, quotes, comments) robustly.
**Prevention:** Always use `python-dotenv`'s `load_dotenv` for loading environment variables from `.env` files. Ensure tests cover edge cases like special characters and comments in configuration values.

## 2026-02-17 - [ProviderRouter SSRF Prevention]
**Vulnerability:** `ProviderRouter` validated URLs against private IPs but failed to resolve hostnames, allowing `localtest.me` (resolving to 127.0.0.1) to bypass checks.
**Learning:** Hostname-based allowlisting/blocklisting is insufficient for SSRF protection due to DNS rebinding and internal domain names. Always resolve the hostname to an IP and validate the IP itself.
**Prevention:** Added DNS resolution using `socket.getaddrinfo` and validated resolved IPs against private/loopback ranges using `ipaddress` library before allowing the request.
