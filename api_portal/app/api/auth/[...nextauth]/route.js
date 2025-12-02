import NextAuth from "next-auth";
import AzureADProvider from "next-auth/providers/azure-ad";
import { signPayload } from "../../../../lib/hmac";

export const authOptions = {
  providers: [
    AzureADProvider({
      clientId: process.env.AZURE_AD_CLIENT_ID,
      clientSecret: process.env.AZURE_AD_CLIENT_SECRET,
      tenantId: process.env.AZURE_AD_TENANT_ID, // can be 'common' for multi-tenant
    }),
  ],
  secret: process.env.NEXTAUTH_SECRET,
  session: {
    strategy: "jwt", // or "database"
  },

  callbacks: {
    async signIn({ user, account, profile }) {
      if (account.provider === "azure-ad") {
        try {
          const backendUrl = process.env.BACKEND_URL;
          const email = user.email;
          const name = user.name;
          const externalId = user.id; // NextAuth normalizes the ID
          const secret = process.env.HMAC_SECRET;

          console.log("[NextAuth] SignIn attempt:", {
            email,
            name,
            externalId,
            backendUrl,
          });

          if (!secret) {
            console.error(
              "[NextAuth] CRITICAL: HMAC_SECRET not set in environment"
            );
            return false;
          }

          if (!backendUrl) {
            console.error(
              "[NextAuth] CRITICAL: BACKEND_URL not set in environment"
            );
            return false;
          }

          console.log(
            `[NextAuth] Secret configured: ${secret.substring(0, 4)}...`
          );
          console.log(`[NextAuth] Backend URL: ${backendUrl}`);

          // Check if user exists
          // For GET request, body is empty string
          const checkHeaders = signPayload(externalId, "", secret);
          const checkUrl = `${backendUrl}/users/by-email/${email}`;

          console.log(`[NextAuth] Checking user existence: ${checkUrl}`);

          const checkRes = await fetch(checkUrl, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              ...checkHeaders,
            },
          });

          console.log(
            `[NextAuth] User check response: ${checkRes.status} ${checkRes.statusText}`
          );

          if (checkRes.status === 404) {
            // Create user
            console.log(
              `[NextAuth] User not found, creating new user: ${email}`
            );

            const body = JSON.stringify({
              email: email,
              name: name,
              external_id: externalId,
            });
            const createHeaders = signPayload(externalId, body, secret);
            const createUrl = `${backendUrl}/users/`;

            console.log(`[NextAuth] Creating user at: ${createUrl}`);

            const createRes = await fetch(createUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...createHeaders,
              },
              body: body,
            });

            const createResponseText = await createRes.text();
            console.log(
              `[NextAuth] Create user response: ${createRes.status} ${createRes.statusText}`
            );
            console.log(`[NextAuth] Create user body:`, createResponseText);

            if (!createRes.ok) {
              console.error(
                "[NextAuth] Failed to create user in backend:",
                createResponseText
              );
              return false; // Deny sign in
            }

            console.log(`[NextAuth] User created successfully: ${email}`);
          } else if (!checkRes.ok) {
            const errorText = await checkRes.text();
            console.error(
              `[NextAuth] Failed to check user existence:`,
              errorText
            );
            console.error(
              `[NextAuth] Status: ${checkRes.status}, URL: ${checkUrl}`
            );
            return false;
          } else {
            console.log(`[NextAuth] User found: ${email}`);
          }

          console.log(`[NextAuth] SignIn successful for: ${email}`);
          return true;
        } catch (error) {
          console.error("[NextAuth] Error in signIn callback:", error);
          console.error("[NextAuth] Error stack:", error.stack);
          return false;
        }
      }
      return true;
    },
    async jwt({ token, user, account }) {
      // If the JWT failed to decrypt, `token` will be missing `.sub`
      // This forces logout when NEXTAUTH_SECRET changes OR cookie is invalid
      if (!token?.sub && !user) {
        return null;
      }

      if (account) {
        token.accessToken = account.access_token;
      }
      if (user) {
        token.id = user.id;
      }

      return token;
    },
    async session({ session, token }) {
      if (!token?.sub) return null; // <--- prevent ghost sessions

      session.accessToken = token.accessToken;
      if (token.id) session.user.id = token.id;

      return session;
    },
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
