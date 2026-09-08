const { Pool } = require('pg');
const path = require('path');

// Load environment variables from root .env
require('dotenv').config({ path: path.resolve(__dirname, '../../../.env') });
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
});

const dbService = {
  // --- 1. USER MANAGEMENT ---
  async getOrCreateUser(username, email) {
    const findQuery = `SELECT user_id FROM users WHERE username = $1;`;
    const existing = await pool.query(findQuery, [username]);
    if (existing.rows.length > 0) {
      return existing.rows[0].user_id;
    }

    const insertQuery = `
      INSERT INTO users (username, email) 
      VALUES ($1, $2) 
      RETURNING user_id;
    `;
    const newUser = await pool.query(insertQuery, [username, email]);
    return newUser.rows[0].user_id;
  },

  // --- 2. CALL SESSION MANAGEMENT ---
  async startCallSession(username, email, callerNumber, receiverNumber, direction, metadata = {}) {
    // Automatically handles old vs new user
    const userId = await this.getOrCreateUser(username, email);

    const query = `
      INSERT INTO user_calls (user_id, caller_number, receiver_number, direction, start_time, status, call_metadata)
      VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, 'completed', $5)
      RETURNING call_id;
    `;
    const values = [userId, callerNumber, receiverNumber, direction, JSON.stringify(metadata)];
    const result = await pool.query(query, values);
    return result.rows[0].call_id;
  },

  async endCallSession(callId, status = 'completed') {
    const query = `
      UPDATE user_calls 
      SET end_time = CURRENT_TIMESTAMP, status = $2
      WHERE call_id = $1
      RETURNING call_id, duration_seconds;
    `;
    const result = await pool.query(query, [callId, status]);
    return result.rows[0];
  },

  // --- 3. INTERACTION LOGGING (Turn-by-Turn) ---
  async logInteraction(callId, userQuestion, aiResponse) {
    const query = `
      INSERT INTO call_interactions (call_id, user_question, ai_response)
      VALUES ($1, $2, $3)
      RETURNING interaction_id, created_at;
    `;
    const values = [callId, userQuestion, aiResponse];
    const result = await pool.query(query, values);
    return result.rows[0];
  },

  // --- 4. RETRIEVAL / EXTRACTION (For AI Context) ---
  async getUserCompleteHistory(username) {
    const query = `
      SELECT 
          uc.call_id,
          uc.direction,
          uc.start_time,
          uc.duration_seconds,
          uc.status,
          uc.call_metadata,
          json_agg(
              json_build_object(
                  'user_question', ci.user_question,
                  'ai_response', ci.ai_response,
                  'time', ci.created_at
              ) ORDER BY ci.created_at ASC
          ) AS interactions
      FROM users u
      JOIN user_calls uc ON u.user_id = uc.user_id
      LEFT JOIN call_interactions ci ON uc.call_id = ci.call_id
      WHERE u.username = $1
      GROUP BY uc.call_id
      ORDER BY uc.start_time DESC;
    `;
    const result = await pool.query(query, [username]);
    return result.rows;
  },

  // Close pool if application shuts down
  async close() {
    await pool.end();
  }
};

module.exports = dbService;