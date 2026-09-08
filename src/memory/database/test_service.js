const dbService = require('./dbService');

async function runTestWorkflow() {
  const testUsername = 'pawan_ganesh';
  const testEmail = 'pawan@vayvora.com';

  try {
    console.log('🚀 Starting Database Service Workflow Test...\n');

    // 1. Simulate starting a call (Automatically checks if user exists or creates a new one)
    console.log('⏳ Initializing call session...');
    const callId = await dbService.startCallSession(
      testUsername,
      testEmail,
      '+919876543210',
      '+18005550199',
      'inbound',
      { source: 'voice_agent_webhook', agent_version: 'v1.0' }
    );
    console.log(`✔️ Call session started successfully. Call ID: ${callId}\n`);

    // 2. Simulate turn-by-turn conversation logging (Happens live while talking)
    console.log('⏳ Logging conversation turns...');
    await dbService.logInteraction(
      callId, 
      'Hi, can you tell me the status of my account update?', 
      'Hello! I can certainly check that for you. Your account update is currently under review.'
    );

    await dbService.logInteraction(
      callId, 
      'How long will the review take?', 
      'It typically takes between 24 to 48 business hours. Is there anything else I can help with?'
    );
    console.log('✔️ Interaction turns logged successfully.\n');

    // 3. Simulate ending the call session
    console.log('⏳ Ending call session...');
    const endResult = await dbService.endCallSession(callId, 'completed');
    console.log(`✔️ Call ended. Duration recorded: ${endResult.duration_seconds || 0} seconds.\n`);

    // 4. Test Extraction: Fetch user's complete history (Used when user calls back later)
    console.log(`⏳ Fetching complete historical records for user: ${testUsername}...`);
    const history = await dbService.getUserCompleteHistory(testUsername);

    console.log(`\n🎉 Test Passed! Found ${history.length} call session(s) for ${testUsername}:`);
    history.forEach((session, index) => {
      console.log(`\n--- Session ${index + 1} (Call ID: ${session.call_id}) ---`);
      console.log(`Date/Time: ${session.start_time} | Status: ${session.status} | Duration: ${session.duration_seconds}s`);
      console.log('Transcript:');
      
      if (session.interactions && session.interactions[0].user_question) {
        session.interactions.forEach((turn, tIdx) => {
          console.log(`  [Turn ${tIdx + 1}]`);
          console.log(`    User: ${turn.user_question}`);
          console.log(`    AI:   ${turn.ai_response}`);
        });
      } else {
        console.log('  (No interactions recorded)');
      }
    });

  } catch (err) {
    console.error('❌ Workflow test failed:', err);
  } finally {
    // Gracefully close database pool connection
    await dbService.close();
  }
}

runTestWorkflow();