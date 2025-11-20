import NextAuth from "next-auth";
import AzureADProvider from "next-auth/providers/azure-ad";

export const authOptions = {
  providers: [
    AzureADProvider({
      clientId: process.env.AZURE_AD_CLIENT_ID,
      clientSecret: process.env.AZURE_AD_CLIENT_SECRET,
      tenantId: process.env.AZURE_AD_TENANT_ID,
    }),
  ],
  session: { strategy: "jwt" },

  callbacks: {
    async jwt({ token, account, profile }) {
      if (account) {
        token.email = profile?.email;
        token.name = profile?.name;
        token.azureId = profile?.sub; // Microsoft unique user ID
      }
      return token;
    },

    async session({ session, token }) {
      session.user.email = token.email;
      session.user.name = token.name;
      session.user.azureId = token.azureId;
      return session;
    },

    async signIn({ profile }) {
      // Restrict to company domain
      return profile.email.endsWith("@bytepulseai.com");
    },
  },

  secret: process.env.AUTH_SECRET,
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
