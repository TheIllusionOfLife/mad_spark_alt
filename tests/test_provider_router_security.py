"""
Tests for ProviderRouter security enhancements.
"""
import socket
import pytest
from unittest.mock import MagicMock, patch

from mad_spark_alt.core.provider_router import ProviderRouter

class TestProviderRouterSecurity:
    """Test security validation in ProviderRouter."""

    @pytest.fixture
    def router(self):
        return ProviderRouter(
            gemini_provider=MagicMock(),
            ollama_provider=MagicMock()
        )

    async def test_validate_url_public_ip(self, router):
        """Test public IP addresses are allowed."""
        # 8.8.8.8 is Google DNS (public)
        await router._validate_url_security("https://8.8.8.8")
        await router._validate_url_security("http://93.184.216.34")  # example.com

    async def test_validate_url_private_ip_blocked(self, router):
        """Test private IP addresses are blocked."""
        private_ips = [
            "http://127.0.0.1",
            "http://localhost",
            "http://10.0.0.1",
            "http://192.168.1.1",
            "http://172.16.0.1",
            "http://[::1]",
            # "http://[fe80::1]", # Parsing literal IPv6 link-local in URL needs [] but urlparse handles hostname
        ]
        for url in private_ips:
            with pytest.raises(ValueError, match="Internal URLs not allowed|Private/internal IP not allowed"):
                await router._validate_url_security(url)

    async def test_validate_url_cloud_metadata_blocked(self, router):
        """Test cloud metadata endpoints are blocked."""
        metadata_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ]
        for url in metadata_urls:
            with pytest.raises(ValueError, match="Cloud metadata endpoints not allowed|Private/internal IP not allowed"):
                await router._validate_url_security(url)

    @patch("socket.getaddrinfo")
    async def test_validate_url_dns_resolution_public(self, mock_getaddrinfo, router):
        """Test hostname resolving to public IP is allowed."""
        # Mock DNS resolution to return a public IP
        # (family, type, proto, canonname, sockaddr)
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
        ]

        await router._validate_url_security("http://example.com")
        mock_getaddrinfo.assert_called()

    @patch("socket.getaddrinfo")
    async def test_validate_url_dns_resolution_private_blocked(self, mock_getaddrinfo, router):
        """Test hostname resolving to private IP is blocked."""
        # Mock DNS resolution to return a private IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.5", 80))
        ]

        with pytest.raises(ValueError, match="resolves to private/internal IP"):
            await router._validate_url_security("http://internal-site.com")

    @patch("socket.getaddrinfo")
    async def test_validate_url_dns_resolution_loopback_blocked(self, mock_getaddrinfo, router):
        """Test hostname resolving to loopback is blocked."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]

        with pytest.raises(ValueError, match="resolves to private/internal IP"):
            await router._validate_url_security("http://localtest.me")

    @patch("socket.getaddrinfo")
    async def test_validate_url_dns_resolution_ipv6_private_blocked(self, mock_getaddrinfo, router):
        """Test hostname resolving to IPv6 private/link-local is blocked."""
        # IPv6 sockaddr is (address, port, flowinfo, scopeid)
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 80, 0, 0))
        ]

        with pytest.raises(ValueError, match="resolves to private/internal IP"):
            await router._validate_url_security("http://ipv6-internal.com")

    @patch("socket.getaddrinfo")
    async def test_validate_url_dns_resolution_failure_blocked(self, mock_getaddrinfo, router):
        """Test DNS resolution failure blocks the request (fail-closed for security)."""
        mock_getaddrinfo.side_effect = socket.gaierror("Name or service not known")

        # Should raise ValueError - fail-closed prevents bypass via DNS failure
        with pytest.raises(ValueError, match="DNS resolution failed"):
            await router._validate_url_security("http://nonexistent-domain.com")
