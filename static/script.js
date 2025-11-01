// Small client helpers for session handling and UI
function getCookie(name) {
  const v = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
  return v ? v.pop() : '';
}
function setCookie(name, value, days=7) {
  const d = new Date(); d.setTime(d.getTime() + days*24*60*60*1000);
  document.cookie = `${name}=${value};path=/;expires=${d.toUTCString()}`;
}
function removeCookie(name) {
  document.cookie = name + "=; Max-Age=0; path=/;";
}

/**
 * Shows a temporary message on the screen.
 * @param {string} text The message to display.
 * @param {'success' | 'error'} type The type of message.
 */
function showMessage(text, type = 'success') {
  let msgEl = document.getElementById('message-popup');
  if (!msgEl) {
    msgEl = document.createElement('div');
    msgEl.id = 'message-popup';
    msgEl.className = 'message';
    document.body.appendChild(msgEl);
  }
  msgEl.textContent = text;
  msgEl.className = `message ${type} show`;
  setTimeout(() => {
    msgEl.className = 'message';
  }, 3000); // Hide after 3 seconds
}


document.addEventListener('DOMContentLoaded', () => {
  // If login form submitted, intercept token response to set cookie
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const data = new URLSearchParams();
      data.append('username', form.username.value);
      data.append('password', form.password.value);

      const res = await fetch('/token', { method: 'POST', body: data });

      if (!res.ok) {
        showMessage('Login failed. Please check your username and password.', 'error');
        return;
      }

      const json = await res.json();
      setCookie('access_token', json.access_token);
      setCookie('user_id', json.user_id);
      setCookie('user_role', json.role);
      window.location.href = '/'; // Redirect to homepage on successful login
    });
  }

  // Register form
  const regForm = document.getElementById('registerForm');
  if (regForm) {
    regForm.addEventListener('submit', async (e) => {
      // FIX: Intercept form submission to handle redirect
      e.preventDefault();
      const form = e.target;
      const data = new FormData(form);

      try {
        const res = await fetch('/api/register', { method: 'POST', body: data });
        const json = await res.json();

        if (!res.ok) {
          throw new Error(json.detail || 'Registration failed');
        }

        // Success! Show message and redirect to login
        showMessage('Registration successful! Please log in.', 'success');
        setTimeout(() => {
          window.location.href = '/login';
        }, 1500); // Wait 1.5s so user can see message

      } catch (err) {
        showMessage(err.message, 'error');
      }
    });
  }

  // Logout
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      removeCookie('access_token'); removeCookie('user_id'); removeCookie('user_role');
      window.location.href = '/';
    });
  }

  // Generic fetch wrapper to include token from cookie
  // This is used for API calls made *within* a page (e.g., liking, commenting)
  window.apiFetch = async (url, opts={}) => {
    opts.headers = opts.headers || {};
    const token = getCookie('access_token');
    if (token) opts.headers['Authorization'] = 'Bearer ' + token;

    const res = await fetch(url, opts);

    if (res.status === 401) {
      // If an API call fails due to auth, redirect to login
      removeCookie('access_token'); removeCookie('user_id'); removeCookie('user_role');
      showMessage('Your session has expired. Please log in again.', 'error');
      setTimeout(() => {
        window.location.href = '/login';
      }, 1500);
      // Return a "faked" response to prevent further errors in the calling code
      return new Response(JSON.stringify({detail: "Not authenticated"}), {status: 401});
    }
    return res;
  };

  // --- Dark Mode Toggle ---
  // Note: This requires a button with id="darkModeToggle" in your HTML
  const toggle = document.getElementById('darkModeToggle');
  const html = document.documentElement;

  // Apply saved theme on load
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    html.classList.add('dark');
  } else if (savedTheme === 'light') {
    html.classList.remove('dark');
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    // Use system preference if no saved theme
    html.classList.add('dark');
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
      } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
      }
    });
  }
});

