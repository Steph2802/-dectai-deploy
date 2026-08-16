// ModerationService.js
//
// This is the single integration point between the DECTAI UI and the
// comment-safety model. Everything else in the app only ever calls
// checkCommentSafety() — swap what happens inside this function and
// nothing else needs to change.
//
// This version calls a local Python API (moderate_api.py) serving a
// model trained on the Jigsaw Toxic Comment dataset. Make sure that
// server is running (python moderate_api.py) before using the app.

/**
 * Check a comment for inappropriate content (profanity, hate speech,
 * harassment, or other flagged content).
 *
 * @param {string} commentText - the raw comment string.
 * @returns {Promise<{ isSafe: boolean, reason?: string }>}
 */
export async function checkCommentSafety(commentText) {
  try {
    const response = await fetch('http://localhost:5000/moderate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: commentText }),
    });

    if (!response.ok) {
      throw new Error(`Moderation API returned ${response.status}`);
    }

    const result = await response.json();
    return { isSafe: result.isSafe, reason: result.reason };
  } catch (err) {
    // If the local model server isn't running, fail safe by blocking
    // the comment and telling you what went wrong, rather than silently
    // posting unchecked content.
    console.error('Moderation API unreachable:', err);
    return {
      isSafe: false,
      reason: 'Could not reach the moderation service. Is moderate_api.py running?',
    };
  }
}
