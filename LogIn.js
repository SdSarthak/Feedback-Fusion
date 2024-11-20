// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyADJSjkj9YZ24DtwLt_ZT9-DnbIBqHKDZ8",
  authDomain: "feedback-fusion-985fc.firebaseapp.com",
  projectId: "feedback-fusion-985fc",
  storageBucket: "feedback-fusion-985fc.firebasestorage.app",
  messagingSenderId: "706594682051",
  appId: "1:706594682051:web:6242049b0593d6b0e91e76",
  measurementId: "G-8EC69YBTC0"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
document.getElementById("LogIn").onsubmit = async (event) => {
  event.preventDefault();
  const email = document.getElementById("username").value;
  const password = document.getElementById("password").value;
  
  try {
      const userCredential = await auth.signInWithEmailAndPassword(email, password);
      window.location.href = "/dashboard"; // Redirect to a protected page
  } catch (error) {
      document.getElementById("errorMessage").innerText = error.message;
  }
};