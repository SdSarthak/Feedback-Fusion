// Sign-in handling for LogIn.html.
//
// Uses the Firebase v9+ modular API. The previous version called
// `auth.signInWithEmailAndPassword(...)` (the v8 namespaced API) on an
// `auth` variable that was never defined, so every submit threw
// "auth is not defined" before it ever reached Firebase.

import {
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence,
  signInWithEmailAndPassword,
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

import { auth, AFTER_AUTH_REDIRECT, friendlyAuthError } from "./firebase-config.js";

const form = document.getElementById("LogIn");
const errorMessage = document.getElementById("errorMessage");
const submitButton = form?.querySelector("button[type='submit']");

function showError(message) {
  if (errorMessage) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  } else {
    alert(message);
  }
}

function clearError() {
  if (errorMessage) {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
  }
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const email = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const remember = document.getElementById("remember")?.checked;

  if (!email || !password) {
    showError("Please enter both an email address and a password.");
    return;
  }

  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "Signing in...";
  }

  try {
    // "Remember me" decides whether the session survives a browser restart.
    await setPersistence(
      auth,
      remember ? browserLocalPersistence : browserSessionPersistence
    );
    await signInWithEmailAndPassword(auth, email, password);
    window.location.href = AFTER_AUTH_REDIRECT;
  } catch (error) {
    showError(friendlyAuthError(error));
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = "Login";
    }
  }
});
