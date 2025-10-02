*** Begin Patch
*** Update File: main.py
@@
 def current_user_email() -> Optional[str]:
     return _session_email() or _iap_email()
 
+def _enrollment_allowed(email: str) -> bool:
+    """
+    Allow access only if the user has at least one registration with
+    enrollment_status = 'accepted'. Superadmin/Admin emails bypass.
+    """
+    if not email:
+        return False
+    try:
+        whitelisted = {str(SUPERADMIN_EMAIL or "").lower(), str(ADMIN_EMAIL or "").lower()}
+        if email.lower() in whitelisted:
+            return True
+        row = fetch_one("""
+            SELECT 1 AS ok
+              FROM public.registrations
+             WHERE lower(user_email) = lower(%s)
+               AND lower(coalesce(enrollment_status,'')) = 'accepted'
+             ORDER BY created_at DESC
+             LIMIT 1;
+        """, (email,))
+        return bool(row and row.get("ok") == 1)
+    except Exception as e:
+        # Fail-closed for regular users; still allow superadmin/admin in emergencies.
+        print(f"[Auth] enrollment check failed for {email}: {e}")
+        return email.lower() in {str(SUPERADMIN_EMAIL or '').lower(), str(ADMIN_EMAIL or '').lower()}
@@
 def _is_public_path(path: str) -> bool:
     if path.startswith(STATIC_URL_PATH):
         return True
     public_exact = {
         "/favicon.ico",
         _bp("/favicon.ico"),
         "/healthz",
         _bp("/healthz"),
         "/login",
         _bp("/login"),
         "/logout",
         _bp("/logout"),
         "/auth/callback",
         _bp("/auth/callback"),
         "/auth/google/callback",
         _bp("/auth/google/callback"),
+        "/access-denied",
+        _bp("/access-denied"),
         "/admin/whoami",
         _bp("/admin/whoami"),
     }
     return path in public_exact
@@
 def enforce_or_attach_identity():
     path = request.path
     if _is_public_path(path):
         return
     email = current_user_email()
     if email:
+        # Block access if enrollment hasn't been accepted yet.
+        if not _enrollment_allowed(email):
+            session.pop("user", None)  # drop session-based identity
+            return redirect(_bp("/access-denied"))
         g.user_email = email
         try:
             g.user_id = ensure_user_row(email)
         except Exception as e:
             print(f"[Auth] ensure_user_row failed for {email}: {e}")
         return
@@
 @app.get("/auth/callback")
 @app.get("/auth/google/callback")
 def auth_callback():
     provider = _require_oauth()
     token = provider.google.authorize_access_token()
@@
     email = (claims.get("email") or "").strip().lower()
     if not email:
         abort(400, description="Google authentication failed (no email).")
 
+    # Enforce enrollment gate BEFORE creating a session.
+    if not _enrollment_allowed(email):
+        # Do not create a session; bounce to an explanatory page.
+        return redirect(_bp("/access-denied"))
+
     session["user"] = {
         "email": email,
         "name": claims.get("name"),
         "picture": claims.get("picture"),
         "sub": claims.get("sub"),
     }
@@
     next_url = _sanitize_next(session.pop("login_next", None))
     return redirect(next_url)
 
+# --- Minimal public page for enrollment-gated users ---
+@app.get("/access-denied")
+def access_denied():
+    msg = (
+        "Access denied: your enrollment is not accepted yet. "
+        "If your status is 'pending', you'll be able to sign in once accepted."
+    )
+    return (msg, 403)
+
+# BASE_PATH alias for access-denied
+if BASE_PATH:
+    app.add_url_rule(f"{BASE_PATH}/access-denied", endpoint="access_denied_bp",
+                     view_func=access_denied, methods=["GET"])
*** End Patch
