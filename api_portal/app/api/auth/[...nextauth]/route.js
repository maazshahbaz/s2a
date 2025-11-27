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
          const backendUrl = process.env.BACKEND_URL || "http://localhost:8001/v1";
          const email = user.email;
          const name = user.name;
          const externalId = user.id; // NextAuth normalizes the ID
          const secret = process.env.HMAC_SECRET;

          if (!secret) {
             console.error("API_KEY_SECRET not set");
             return false;
          }

          // Check if user exists
          // For GET request, body is empty string
          const checkHeaders = signPayload(externalId, "", secret);
          
          const checkRes = await fetch(`${backendUrl}/users/by-email/${email}`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              ...checkHeaders
            },
          });

          if (checkRes.status === 404) {
            // Create user
            const body = JSON.stringify({
                email: email,
                name: name,
                external_id: externalId,
            });
            const createHeaders = signPayload(externalId, body, secret);

            const createRes = await fetch(`${backendUrl}/users/`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...createHeaders
              },
              body: body,
            });

            if (!createRes.ok) {
              console.error("Failed to create user in backend", await createRes.text());
              return false; // Deny sign in
            }
          } else if (!checkRes.ok) {
            console.error("Failed to check user existence", await checkRes.text());
            return false;
          }
          
          return true;
        } catch (error) {
          console.error("Error in signIn callback:", error);
          return false;
        }
      }
      return true;
    },
    async jwt({ token, account, user }) {
      // Store access token in JWT
      if (account) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      return session;
    },
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
