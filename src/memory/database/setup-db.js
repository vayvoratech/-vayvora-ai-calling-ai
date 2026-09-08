const { Pool } = require('pg');
const path = require('path');

// Explicitly load the .env file from the root project directory (-vayvora-ai-calling-ai/.env)
require('dotenv').config({ path: path.resolve(__dirname, '../../../.env') });
// Debug check to verify environment variables are loading correctly
console.log('--- ENV CHECK ---');
console.log('DB_HOST:', process.env.DB_HOST);
console.log('DB_USER:', process.env.DB_USER);
console.log('DB_NAME:', process.env.DB_NAME);
console.log('DB_PASSWORD length:', process.env.DB_PASSWORD ? process.env.DB_PASSWORD.length : 'UNDEFINED');
console.log('-----------------');

const pool = new Pool({
  user: process.env.DB_USER || 'postgres',
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'ai_call_db',
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT || 5432,
});

async function initializeDatabaseSchema() {
  const client = await pool.connect();
  try {
    console.log('⏳ Starting database schema creation...');

    // Begin transaction
    await client.query('BEGIN');

    // 1. Enable UUID extension if not already present
    await client.query(`CREATE EXTENSION IF NOT EXISTS "pgcrypto";`);

    // 2. Create users table
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
          user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          username VARCHAR(50) UNIQUE NOT NULL,
          email VARCHAR(100) UNIQUE NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `);
    console.log('✔️ Checked/Created table: users');

    // 3. Create user_calls table
    await client.query(`
      CREATE TABLE IF NOT EXISTS user_calls (
          call_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
          caller_number VARCHAR(20) NOT NULL,
          receiver_number VARCHAR(20) NOT NULL,
          direction VARCHAR(10) CHECK (direction IN ('inbound', 'outbound')),
          start_time TIMESTAMP WITH TIME ZONE NOT NULL,
          end_time TIMESTAMP WITH TIME ZONE,
          duration_seconds INTEGER GENERATED ALWAYS AS (
              EXTRACT(EPOCH FROM (end_time - start_time))::INTEGER
          ) STORED,
          status VARCHAR(20) CHECK (status IN ('completed', 'missed', 'busy', 'failed')),
          call_metadata JSONB,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `);
    console.log('✔️ Checked/Created table: user_calls');

    // 4. Create call_interactions table
    await client.query(`
      CREATE TABLE IF NOT EXISTS call_interactions (
          interaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          call_id UUID REFERENCES user_calls(call_id) ON DELETE CASCADE,
          user_question TEXT NOT NULL,
          ai_response TEXT NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `);
    console.log('✔️ Checked/Created table: call_interactions');

    // 5. Create performance indexes
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_user_calls_user_id ON user_calls(user_id);
      CREATE INDEX IF NOT EXISTS idx_user_calls_start_time ON user_calls(start_time);
      CREATE INDEX IF NOT EXISTS idx_call_interactions_call_id ON call_interactions(call_id);
    `);
    console.log('✔️ Checked/Created performance indexes');

    // Commit transaction
    await client.query('COMMIT');
    console.log('🎉 Database schema successfully established and verified!');
  } catch (err) {
    await client.query('ROLLBACK');
    console.error('❌ Error executing schema setup script:', err);
  } finally {
    client.release();
    await pool.end();
  }
}

initializeDatabaseSchema();