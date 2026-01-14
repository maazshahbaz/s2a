/*
  Warnings:

  - The primary key for the `auth_keys` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `key_hash` on the `auth_keys` table. All the data in the column will be lost.
  - You are about to drop the column `key_id` on the `auth_keys` table. All the data in the column will be lost.
  - The `id` column on the `auth_keys` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - A unique constraint covering the columns `[key]` on the table `auth_keys` will be added. If there are existing duplicate values, this will fail.
  - A unique constraint covering the columns `[hash]` on the table `auth_keys` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `hash` to the `auth_keys` table without a default value. This is not possible if the table is not empty.
  - The required column `key` was added to the `auth_keys` table with a prisma-level default value. This is not possible if the table is not empty. Please add this column as optional, then populate it before making it required.
  - Added the required column `user_id` to the `auth_keys` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "public"."auth_keys_key_hash_key";

-- DropIndex
DROP INDEX "public"."auth_keys_key_id_key";

-- AlterTable
ALTER TABLE "public"."auth_keys" DROP CONSTRAINT "auth_keys_pkey",
DROP COLUMN "key_hash",
DROP COLUMN "key_id",
ADD COLUMN     "hash" TEXT NOT NULL,
ADD COLUMN     "key" TEXT NOT NULL,
ADD COLUMN     "user_id" INTEGER NOT NULL,
DROP COLUMN "id",
ADD COLUMN     "id" SERIAL NOT NULL,
ADD CONSTRAINT "auth_keys_pkey" PRIMARY KEY ("id");

-- CreateTable
CREATE TABLE "public"."User" (
    "id" SERIAL NOT NULL,
    "key" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "name" TEXT,
    "externalId" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."EmailOtp" (
    "id" SERIAL NOT NULL,
    "user_id" INTEGER NOT NULL,
    "otp_hash" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "consumed" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "EmailOtp_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."RefreshToken" (
    "id" SERIAL NOT NULL,
    "user_id" INTEGER NOT NULL,
    "token_hash" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "revoked" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "RefreshToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_key_key" ON "public"."User"("key");

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "public"."User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "User_externalId_key" ON "public"."User"("externalId");

-- CreateIndex
CREATE UNIQUE INDEX "EmailOtp_user_id_key" ON "public"."EmailOtp"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "auth_keys_key_key" ON "public"."auth_keys"("key");

-- CreateIndex
CREATE UNIQUE INDEX "auth_keys_hash_key" ON "public"."auth_keys"("hash");

-- AddForeignKey
ALTER TABLE "public"."EmailOtp" ADD CONSTRAINT "EmailOtp_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."RefreshToken" ADD CONSTRAINT "RefreshToken_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."auth_keys" ADD CONSTRAINT "auth_keys_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
