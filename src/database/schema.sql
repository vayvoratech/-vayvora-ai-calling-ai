-- Schema definition for Vayvora AI Voice Calling PostgreSQL Database
-- Verified against live database instance on port 5434

CREATE TABLE IF NOT EXISTS leads (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    phone_number VARCHAR NOT NULL,
    email VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'New',
    interested_in TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT leads_phone_number_key UNIQUE (phone_number)
);

CREATE INDEX IF NOT EXISTS idx_leads_phone ON leads USING btree (phone_number);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads USING btree (status);

CREATE TABLE IF NOT EXISTS call_interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    speaker VARCHAR NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_call_interactions_session ON call_interactions USING btree (session_id);

CREATE TABLE IF NOT EXISTS hr_followup_queue (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    phone_number VARCHAR NOT NULL,
    email VARCHAR,
    reason TEXT NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'queued',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_hr_queue_created ON hr_followup_queue USING btree (created_at DESC);

CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER REFERENCES leads(id) ON DELETE SET NULL,
    phone_number VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    event_date VARCHAR NOT NULL,
    event_time VARCHAR NOT NULL,
    slot VARCHAR NOT NULL,
    project_name VARCHAR,
    external_event_id VARCHAR,
    description TEXT,
    attendee_name VARCHAR,
    attendee_email VARCHAR,
    timezone VARCHAR NOT NULL DEFAULT 'Asia/Kolkata',
    status VARCHAR NOT NULL DEFAULT 'confirmed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_appointments_phone ON appointments USING btree (phone_number);
CREATE INDEX IF NOT EXISTS idx_appointments_lead_id ON appointments USING btree (lead_id);
CREATE INDEX IF NOT EXISTS idx_appointments_external_id ON appointments USING btree (external_event_id);
CREATE INDEX IF NOT EXISTS idx_appointments_status ON appointments USING btree (status);

CREATE TABLE IF NOT EXISTS appointment_reconciliations (
    id SERIAL PRIMARY KEY,
    appointment_id INTEGER,
    external_event_id VARCHAR,
    phone_number VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    details TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_appointment_reconciliations_phone ON appointment_reconciliations USING btree (phone_number);
