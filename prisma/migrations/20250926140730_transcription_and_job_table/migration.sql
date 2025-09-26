-- CreateTable
CREATE TABLE "public"."transcription_jobs" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "job_id" TEXT NOT NULL,
    "audio_path" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "is_async" BOOLEAN NOT NULL,
    "enhance_audio" BOOLEAN NOT NULL DEFAULT true,
    "remove_silence" BOOLEAN NOT NULL DEFAULT false,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "callback_url" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "audio_duration" DOUBLE PRECISION,
    "error_message" TEXT,

    CONSTRAINT "transcription_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."transcription_results" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "job_id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "confidence" DOUBLE PRECISION,
    "rtf" DOUBLE PRECISION,
    "processing_time" DOUBLE PRECISION,
    "chunks" INTEGER DEFAULT 1,
    "audio_quality" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "transcription_results_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "transcription_jobs_job_id_key" ON "public"."transcription_jobs"("job_id");

-- CreateIndex
CREATE UNIQUE INDEX "transcription_results_job_id_key" ON "public"."transcription_results"("job_id");

-- AddForeignKey
ALTER TABLE "public"."transcription_results" ADD CONSTRAINT "transcription_results_job_id_fkey" FOREIGN KEY ("job_id") REFERENCES "public"."transcription_jobs"("job_id") ON DELETE RESTRICT ON UPDATE CASCADE;
