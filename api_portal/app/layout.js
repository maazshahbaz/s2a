import "./globals.css";
import { Providers } from "./providers";
import { getServerSession } from "next-auth";
import { authOptions } from "./api/auth/[...nextauth]/route";

export default async function RootLayout({ children }) {
  const session = await getServerSession(authOptions);

  return (
    <html lang="en">
      <body>
        <header className="p-4 bg-white dark:bg-gray-800 shadow-sm">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            S2A Portal
          </h1>
        </header>
        <main className="container fade-in">
          <Providers serverSession={session}>{children}</Providers>
        </main>
        <footer className="p-4 text-center text-sm text-gray-600 dark:text-gray-400">
          © {new Date().getFullYear()} S2A
        </footer>
      </body>
    </html>
  );
}
