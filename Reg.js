// Registration handling for Reg.html.
//
// The previous version called signInWithEmailAndPassword on the registration
// form, so it tried to log people in to accounts that did not exist yet. It
// also used the v8 namespaced API on an undefined `auth` object. This now
// creates the account with the v9+ modular API and stores the profile
// details the form collects.

import {
  createUserWithEmailAndPassword,
  updateProfile,
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

import { auth, AFTER_AUTH_REDIRECT, friendlyAuthError } from "./firebase-config.js";

const form = document.getElementById("Regform");
const errorMessage = document.getElementById("errorMessage");
const submitButton = form?.querySelector("button");

function value(id) {
  return document.getElementById(id)?.value?.trim() ?? "";
}

function showError(message) {
  if (errorMessage) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
    errorMessage.scrollIntoView({ behavior: "smooth", block: "center" });
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

function selectedGender() {
  return document.querySelector("input[name='gender']:checked")?.value || "";
}

/** Persist the extra profile fields, if Firestore is set up for the project. */
async function saveProfile(uid, profile) {
  try {
    const { getFirestore, doc, setDoc } = await import(
      "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js"
    );
    await setDoc(doc(getFirestore(), "users", uid), profile);
    return true;
  } catch (error) {
    // Firestore may not be enabled, or rules may block the write. The
    // account itself is already created, so registration still succeeds.
    console.warn("Could not save extended profile:", error);
    return false;
  }
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const fullName = value("fullname");
  const email = value("username");
  const password = document.getElementById("password")?.value ?? "";
  const confirmPassword = document.getElementById("confirmPassword")?.value ?? "";

  if (!fullName || !email || !password) {
    showError("Please fill in your name, email address and password.");
    return;
  }
  if (password.length < 6) {
    showError("Password must be at least 6 characters long.");
    return;
  }
  if (password !== confirmPassword) {
    showError("The two passwords do not match.");
    return;
  }

  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "Creating account...";
  }

  try {
    const credential = await createUserWithEmailAndPassword(auth, email, password);
    await updateProfile(credential.user, { displayName: fullName });

    await saveProfile(credential.user.uid, {
      fullName,
      email,
      phone: value("phone"),
      birthDate: value("birthdate"),
      gender: selectedGender(),
      address: value("address"),
      country: value("country"),
      city: value("city"),
      region: value("region"),
      postalCode: value("postalcode"),
      createdAt: new Date().toISOString(),
    });

    window.location.href = AFTER_AUTH_REDIRECT;
  } catch (error) {
    showError(friendlyAuthError(error));
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = "Submit";
    }
  }
});
