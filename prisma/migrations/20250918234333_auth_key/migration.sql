-- CreateTable
CREATE TABLE "public"."auth_keys" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "key_id" TEXT NOT NULL,
    "key_hash" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "key_type" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_used" TIMESTAMP(3),
    "usage_count" INTEGER NOT NULL DEFAULT 0,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "requests_per_minute" INTEGER NOT NULL DEFAULT 60,
    "requests_per_hour" INTEGER NOT NULL DEFAULT 1000,
    "requests_per_day" INTEGER NOT NULL DEFAULT 10000,
    "total_audio_minutes" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_requests" INTEGER NOT NULL DEFAULT 0,
    "permissions" TEXT[],

    CONSTRAINT "auth_keys_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "auth_keys_key_id_key" ON "public"."auth_keys"("key_id");

-- CreateIndex
CREATE UNIQUE INDEX "auth_keys_key_hash_key" ON "public"."auth_keys"("key_hash");
