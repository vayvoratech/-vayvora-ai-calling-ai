Vayvora Technology — Legal & Policies
Legal Information

Vayvora Technology Pvt Ltd provides several legal and policy documents through its official website.

The currently available company information identifies the following documents:

Terms & Conditions
Privacy Policy
Workplace Policy

The actual contents and provisions of these documents have not been provided in the current knowledge base.

Terms & Conditions

Vayvora Technology has a Terms & Conditions page on its official website.

The current knowledge base does not contain the actual terms.

The AI voice agent must not:

Interpret specific contractual clauses
State specific contractual obligations
Confirm liability terms
Confirm cancellation terms
Confirm refund terms
Confirm warranty terms
Confirm intellectual-property terms
Confirm dispute-resolution procedures
Confirm governing law
Provide legal advice

unless the approved Terms & Conditions document has been added to the knowledge base.

Privacy Policy

Vayvora Technology has a Privacy Policy page on its official website.

The actual privacy-policy content has not been provided in the current knowledge base.

The AI voice agent must not make unsupported claims about:

Personal-data collection
Personal-data storage
Data retention
Data deletion
Cookies
Third-party data sharing
Data processing
User rights
Security practices
International data transfers
Privacy compliance

unless these details are available in the verified Privacy Policy document.

Workplace Policy

Vayvora Technology has a Workplace Policy page on its official website.

The actual contents of the Workplace Policy have not been provided in the current knowledge base.

The AI voice agent must not describe specific workplace rules, employee obligations, disciplinary procedures, leave policies, conduct requirements, or other workplace provisions unless the official document is added to the knowledge base.

Legal Advice Restriction

The AI voice agent should not provide legal advice.

If a caller asks the agent to interpret a legal document or determine the caller's legal rights or obligations, the agent should explain that it can provide information from the company's approved documents but cannot provide legal advice.

Legal Document Access

When the actual legal documents are added to the RAG knowledge base, each document should be ingested as a separate source.

Recommended structure:

legal/
├── terms_conditions.md
├── privacy_policy.md
└── workplace_policy.md

Keeping the documents separate allows the retrieval system to identify the correct policy source.

Voice Agent Response Rules
Caller: "What are your terms and conditions?"

If the actual Terms & Conditions document is not available in the knowledge base, the agent should state that the detailed terms are not currently available to it and direct the caller to the official Terms & Conditions page.

Caller: "What is your privacy policy?"

The agent can confirm that Vayvora Technology has a Privacy Policy, but should not describe specific provisions unless the policy content has been added to the knowledge base.

Caller: "Do you sell customer data?"

The agent should not answer based on assumptions.

It should retrieve the official Privacy Policy if available. If the policy is not available, it should state that the detailed privacy information is not currently available in the knowledge base.

Caller: "Can you explain this contract?"

The agent should not provide legal interpretation. It can summarize an approved company document if that document has been added to the knowledge base, while making clear that the summary is informational and not legal advice.

Caller: "What is your refund policy?"

No official refund policy has been provided in the current knowledge base.

The agent must not invent a refund policy.

Information Required for Complete Legal Knowledge

The following official documents should eventually be added:

Complete Terms & Conditions
Complete Privacy Policy
Complete Workplace Policy
Refund Policy
Cancellation Policy
Cookie Policy, if applicable
Data Processing Agreement, if applicable
Service Agreement
SLA documents
Warranty terms
Intellectual Property terms
Confidentiality/NDA terms

Only verified company-approved versions should be added.

Document Versioning

Legal documents are subject to change.

The production RAG system should retain metadata such as:

Document name
Document version
Effective date
Last updated date
Source URL
Approval status

The latest approved version should be preferred during retrieval.

Critical RAG Rule

Legal information must always come from verified, approved company documents.

The AI voice agent must never fabricate, interpret, or assume legal terms.

If the required legal information is unavailable, the agent should direct the caller to the appropriate official document or Vayvora Technology representative.

Source Scope

This document records the currently known legal and policy pages for Vayvora Technology.

It does not reproduce or infer the contents of those legal documents.