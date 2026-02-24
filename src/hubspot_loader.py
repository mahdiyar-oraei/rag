"""HubSpot CRM data loader for the RAG pipeline."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import hubspot
from hubspot.crm.companies import ApiException as CompaniesApiException
from hubspot.crm.contacts import ApiException as ContactsApiException
from hubspot.crm.deals import ApiException as DealsApiException
from hubspot.crm.owners import ApiException as OwnersApiException
from langchain_core.documents import Document

from .config import HUBSPOT_ACCESS_TOKEN, HUBSPOT_BASE_URL

_CONTACT_PROPS = ["firstname", "lastname", "email", "phone", "company", "jobtitle"]
_COMPANY_PROPS = ["name", "domain", "industry", "city", "state", "country", "phone"]
_DEAL_PROPS = ["dealname", "dealstage", "amount", "closedate", "pipeline"]
_PAGE_LIMIT = 100
_API_TIMEOUT = 600  # seconds per API call
_EU_BASE_URL = "https://api-eu1.hubapi.com"


class HubSpotLoader:
    """Fetch CRM records from HubSpot and convert them to LangChain Documents."""

    def __init__(self) -> None:
        if not HUBSPOT_ACCESS_TOKEN:
            raise ValueError(
                "HUBSPOT_ACCESS_TOKEN is not set. Add it to your .env file."
            )
        # EU tokens (pat-eu1-...) must use the EU API endpoint
        config: dict = {"access_token": HUBSPOT_ACCESS_TOKEN}
        if HUBSPOT_BASE_URL:
            config["host"] = HUBSPOT_BASE_URL.rstrip("/")
        elif HUBSPOT_ACCESS_TOKEN.strip().lower().startswith("pat-eu1"):
            config["host"] = _EU_BASE_URL
        self.client = hubspot.Client.create(**config)

    def _run_with_timeout(self, fn, *args, **kwargs):
        """Run a callable with a timeout to avoid indefinite hangs."""
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=_API_TIMEOUT)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"HubSpot API call timed out after {_API_TIMEOUT}s. "
                    "Check your token and network, or try again."
                ) from None

    # ------------------------------------------------------------------
    # Individual object loaders
    # ------------------------------------------------------------------

    def _fetch_contacts(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all contacts and return as Documents."""
        docs: list[Document] = []
        after: str | None = None

        while True:
            try:
                page = self.client.crm.contacts.basic_api.get_page(
                    limit=_PAGE_LIMIT,
                    after=after,
                    properties=_CONTACT_PROPS,
                )
            except ContactsApiException as exc:
                raise RuntimeError(f"HubSpot Contacts API error: {exc}") from exc

            for record in page.results:
                props = record.properties or {}
                first = props.get("firstname") or ""
                last = props.get("lastname") or ""
                name = f"{first} {last}".strip() or "Unknown"
                lines = [
                    f"Contact: {name}",
                    f"Email: {props.get('email') or 'N/A'}",
                    f"Phone: {props.get('phone') or 'N/A'}",
                    f"Company: {props.get('company') or 'N/A'}",
                    f"Job Title: {props.get('jobtitle') or 'N/A'}",
                ]
                docs.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source": "hubspot",
                            "object_type": "contact",
                            "hs_object_id": record.id,
                        },
                    )
                )

            if on_progress:
                on_progress("contacts", len(docs))
            after = None
            if page.paging and page.paging.next:
                after = page.paging.next.after
            if not after:
                break

        return docs

    def load_contacts(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all contacts (with timeout when on_progress is not used)."""
        if on_progress is not None:
            return self._fetch_contacts(on_progress=on_progress)
        return self._run_with_timeout(self._fetch_contacts)

    def _fetch_companies(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all companies and return as Documents."""
        docs: list[Document] = []
        after: str | None = None

        while True:
            try:
                page = self.client.crm.companies.basic_api.get_page(
                    limit=_PAGE_LIMIT,
                    after=after,
                    properties=_COMPANY_PROPS,
                )
            except CompaniesApiException as exc:
                raise RuntimeError(f"HubSpot Companies API error: {exc}") from exc

            for record in page.results:
                props = record.properties or {}
                lines = [
                    f"Company: {props.get('name') or 'Unknown'}",
                    f"Domain: {props.get('domain') or 'N/A'}",
                    f"Industry: {props.get('industry') or 'N/A'}",
                    f"Location: {', '.join(filter(None, [props.get('city'), props.get('state'), props.get('country')]))}",
                    f"Phone: {props.get('phone') or 'N/A'}",
                ]
                docs.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source": "hubspot",
                            "object_type": "company",
                            "hs_object_id": record.id,
                        },
                    )
                )

            if on_progress:
                on_progress("companies", len(docs))
            after = None
            if page.paging and page.paging.next:
                after = page.paging.next.after
            if not after:
                break

        return docs

    def load_companies(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all companies (with timeout when on_progress is not used)."""
        if on_progress is not None:
            return self._fetch_companies(on_progress=on_progress)
        return self._run_with_timeout(self._fetch_companies)

    def _get_association_id(
        self, assoc: object, key: str
    ) -> str | None:
        """Extract first associated object ID from associations."""
        if not assoc:
            return None
        data = assoc.get(key) if isinstance(assoc, dict) else getattr(assoc, key, None)
        if not data:
            return None
        results = getattr(data, "results", None) or (
            data.get("results", []) if isinstance(data, dict) else []
        )
        if not results:
            return None
        first = results[0]
        return str(getattr(first, "id", first.get("id") if isinstance(first, dict) else None))

    def _fetch_deals(
        self,
        on_progress: Callable[[str, int], None] | None = None,
        company_map: dict[str, str] | None = None,
    ) -> list[Document]:
        """Fetch all deals and return as Documents. Enrich with company name and contact associations."""
        docs: list[Document] = []
        after: str | None = None
        company_map = company_map or {}

        while True:
            try:
                kwargs: dict = {
                    "limit": _PAGE_LIMIT,
                    "properties": _DEAL_PROPS,
                    "associations": ["companies", "contacts"],
                }
                if after:
                    kwargs["after"] = after
                page = self.client.crm.deals.basic_api.get_page(**kwargs)
            except DealsApiException as exc:
                raise RuntimeError(f"HubSpot Deals API error: {exc}") from exc

            for record in page.results:
                props = record.properties or {}
                amount = props.get("amount")
                amount_str = f"${float(amount):,.2f}" if amount else "N/A"
                lines = [
                    f"Deal: {props.get('dealname') or 'Unknown'}",
                    f"Stage: {props.get('dealstage') or 'N/A'}",
                    f"Pipeline: {props.get('pipeline') or 'N/A'}",
                    f"Amount: {amount_str}",
                    f"Close Date: {props.get('closedate') or 'N/A'}",
                ]
                assoc = getattr(record, "associations", None)
                company_name = "N/A"
                if company_map and assoc:
                    cid = self._get_association_id(assoc, "companies")
                    if cid:
                        company_name = company_map.get(str(cid), "N/A")
                lines.append(f"Company: {company_name}")

                contact_id = None
                if assoc:
                    contact_id = self._get_association_id(assoc, "contacts")

                meta: dict = {
                    "source": "hubspot",
                    "object_type": "deal",
                    "hs_object_id": record.id,
                }
                if contact_id:
                    meta["associated_contact_id"] = contact_id

                docs.append(Document(page_content="\n".join(lines), metadata=meta))

            if on_progress:
                on_progress("deals", len(docs))
            after = None
            if page.paging and page.paging.next:
                after = page.paging.next.after
            if not after:
                break

        return docs

    def load_deals(
        self,
        on_progress: Callable[[str, int], None] | None = None,
        companies: list[Document] | None = None,
    ) -> list[Document]:
        """Fetch all deals (with timeout when on_progress is not used). Enrich with company names when companies provided."""
        company_map: dict[str, str] = {}
        if companies:
            for doc in companies:
                cid = doc.metadata.get("hs_object_id")
                if cid:
                    first_line = doc.page_content.split("\n")[0]
                    name = first_line.replace("Company: ", "").strip() or "Unknown"
                    company_map[str(cid)] = name
        if on_progress is not None:
            return self._fetch_deals(on_progress=on_progress, company_map=company_map or None)
        return self._run_with_timeout(
            lambda: self._fetch_deals(company_map=company_map or None)
        )

    def _fetch_owners(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all owners (sales reps) and return as Documents."""
        docs: list[Document] = []
        after: str | None = None

        while True:
            try:
                page = self.client.crm.owners.owners_api.get_page(
                    limit=_PAGE_LIMIT,
                    after=after,
                )
            except OwnersApiException as exc:
                raise RuntimeError(f"HubSpot Owners API error: {exc}") from exc

            for record in page.results:
                name = f"{record.first_name or ''} {record.last_name or ''}".strip() or "Unknown"
                lines = [
                    f"Owner: {name}",
                    f"Email: {record.email or 'N/A'}",
                ]
                docs.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source": "hubspot",
                            "object_type": "owner",
                            "hs_object_id": str(record.id),
                        },
                    )
                )

            if on_progress:
                on_progress("owners", len(docs))
            after = None
            if page.paging and page.paging.next:
                after = page.paging.next.after
            if not after:
                break

        return docs

    def load_owners(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> list[Document]:
        """Fetch all owners (with timeout when on_progress is not used)."""
        if on_progress is not None:
            return self._fetch_owners(on_progress=on_progress)
        return self._run_with_timeout(self._fetch_owners)

    # ------------------------------------------------------------------
    # Aggregate loader
    # ------------------------------------------------------------------

    def load_all(
        self,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> tuple[list[Document], dict[str, int]]:
        """
        Fetch all CRM objects and return combined Documents with a count summary.
        Deals are enriched with company names when available.

        Returns:
            Tuple of (all_documents, counts_per_object_type).
        """
        all_docs: list[Document] = []
        counts: dict[str, int] = {}

        contacts = self.load_contacts(on_progress=on_progress)
        counts["contacts"] = len(contacts)
        all_docs.extend(contacts)

        companies = self.load_companies(on_progress=on_progress)
        counts["companies"] = len(companies)
        all_docs.extend(companies)

        deals = self.load_deals(on_progress=on_progress, companies=companies)
        counts["deals"] = len(deals)
        all_docs.extend(deals)

        owners = self.load_owners(on_progress=on_progress)
        counts["owners"] = len(owners)
        all_docs.extend(owners)

        return all_docs, counts
