// Firebase web configuration.
//
// These values are NOT secrets. Firebase web API keys are shipped to every
// browser that loads the app and are safe to commit - they identify the
// project, they do not grant access. Real protection comes from:
//   1. Firebase Authentication (who can sign in)
//   2. Firestore / Storage Security Rules (what they can read and write)
//   3. the Authorized Domains list in the Firebase console
// See https://firebase.google.com/docs/projects/api-keys
//
// To point the frontend at your own Firebase project, replace the values
// below with the config from Project Settings > Your apps > Web app.

import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

export const firebaseConfig = {
  apiKey: "AIzaSyADJSjkj9YZ24DtwLt_ZT9-DnbIBqHKDZ8",
  authDomain: "feedback-fusion-985fc.firebaseapp.com",
  projectId: "feedback-fusion-985fc",
  storageBucket: "feedback-fusion-985fc.firebasestorage.app",
  messagingSenderId: "706594682051",
  appId: "1:706594682051:web:6242049b0593d6b0e91e76",
  measurementId: "G-8EC69YBTC0",
};

export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Where users land after a successful sign in or registration.
export const AFTER_AUTH_REDIRECT = "Feedback Fusion.html";

/** Turn a Firebase auth error code into something a human can act on. */
export function friendlyAuthError(error) {
  const messages = {
    "auth/invalid-email": "That email address is not valid.",
    "auth/user-disabled": "This account has been disabled.",
    "auth/user-not-found": "No account exists for that email address.",
    "auth/wrong-password": "Incorrect password. Please try again.",
    "auth/invalid-credential": "Incorrect email or password.",
    "auth/email-already-in-use": "An account already exists for that email.",
    "auth/weak-password": "Password must be at least 6 characters long.",
    "auth/too-many-requests": "Too many attempts. Please wait and try again.",
    "auth/network-request-failed": "Network error. Check your connection.",
    "auth/operation-not-allowed":
      "Email/password sign-in is not enabled for this Firebase project.",
  };
  return messages[error?.code] || error?.message || "Something went wrong.";
}
