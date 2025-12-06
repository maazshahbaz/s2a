/*
  Warnings:

  - You are about to drop the column `completed_at` on the `transcription_jobs` table. All the data in the column will be lost.
  - You are about to drop the column `started_at` on the `transcription_jobs` table. All the data in the column will be lost.
  - You are about to drop the column `processing_time` on the `transcription_results` table. All the data in the column will be lost.

*/
-- Step 1: Add new columns to transcription_jobs
ALTER TABLE "transcription_jobs" 
ADD COLUMN "audio_size" BIGINT,
ADD COLUMN "processing_time" DOUBLE PRECISION;

-- Step 2: Migrate processing_time data from transcription_results to transcription_jobs
UPDATE "transcription_jobs" tj
SET "processing_time" = tr."processing_time"
FROM "transcription_results" tr
WHERE tj."job_id" = tr."job_id";

-- Step 3: Drop old columns from transcription_jobs
ALTER TABLE "transcription_jobs" 
DROP COLUMN "completed_at",
DROP COLUMN "started_at";

-- Step 4: Drop processing_time from transcription_results
ALTER TABLE "transcription_results" DROP COLUMN "processing_time";
