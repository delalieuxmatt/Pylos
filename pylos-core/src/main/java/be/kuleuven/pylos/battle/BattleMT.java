package be.kuleuven.pylos.battle;

import be.kuleuven.pylos.player.PylosPlayerType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger; // NEW: Import thread-safe counter
import java.util.stream.Collectors;

public class BattleMT {

    static int N_RUNS_PER_JOB = 2; //has to be even

    public static BattleResult play(PylosPlayerType p1, PylosPlayerType p2, int runs, int nThreads) {
        return play(p1, p2, runs, nThreads, true);
    }

    public static BattleResult play(PylosPlayerType p1, PylosPlayerType p2, int runs, int nThreads, boolean print) {
        int nTasks = runs / N_RUNS_PER_JOB;
        int rest = runs % N_RUNS_PER_JOB;

        // --- NEW: Create a shared, thread-safe counter ---
        final AtomicInteger gameCounter = new AtomicInteger(0);
        final int totalRuns = runs;
        // --------------------------------------------------

        ExecutorService pool = Executors.newFixedThreadPool(nThreads);
        List<BattleRunnable> battleRunnables = new ArrayList<>();

        for (int i = 0; i < nTasks; i++) {
            // MODIFIED: Pass the counter and total to the runnable
            BattleRunnable r = new BattleRunnable(p1, p2, N_RUNS_PER_JOB, gameCounter, totalRuns);
            battleRunnables.add(r);
            pool.execute(r);
        }

        if (rest > 0) {
            // MODIFIED: Pass the counter and total to the runnable
            BattleRunnable r = new BattleRunnable(p1, p2, rest, gameCounter, totalRuns);
            battleRunnables.add(r);
            pool.execute(r);
        }
        pool.shutdown();

        try {
            //wait until the pool has processed all games
            long timeout = runs * 10L; //timeout of 10 seconds per game
            pool.awaitTermination(timeout, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        if (battleRunnables.stream().anyMatch(r -> r.result == null)) {
            throw new RuntimeException("Not all battles were completed!");
        }

        BattleResult result = BattleResult.merge(battleRunnables.stream().map(r -> r.result).collect(Collectors.toList()));

        if (print) {
            result.print();
        }

        return result;
    }

    private static class BattleRunnable implements Runnable {
        private final PylosPlayerType p1;
        private final PylosPlayerType p2;
        private final int nRuns;
        private BattleResult result;

        // --- NEW: Add fields for the counter and total ---
        private final AtomicInteger gameCounter;
        private final int totalRuns;
        // -----------------------------------------------

        // MODIFIED: Update constructor
        public BattleRunnable(PylosPlayerType p1, PylosPlayerType p2, int nRuns, AtomicInteger gameCounter, int totalRuns) {
            this.p1 = p1;
            this.p2 = p2;
            this.nRuns = nRuns;
            this.gameCounter = gameCounter; // NEW
            this.totalRuns = totalRuns;     // NEW
        }

        // MODIFIED: Add logging
        @Override
        public void run() {
            try {
                // This runs the small batch of games (e.g., 2)
                result = Battle.play(p1, p2, nRuns, false);

                // --- NEW: Logging logic ---
                // Atomically add the number of games we just ran to the total count
                int previousGames = gameCounter.get();
                int completedGames = gameCounter.addAndGet(nRuns);

                // Check if this batch crossed a 1000-game milestone
                // This is more robust than (completedGames % 1000 == 0)
                if ((completedGames / 100) > (previousGames / 100)) {
                    double percentage = (double) completedGames / totalRuns * 100.0;
                    String timestamp = java.time.LocalTime.now().toString();

                    // Print to System.err for immediate, unbuffered output
                    System.err.printf("[%s] --- Processed ~%d / %d games (%.1f%%)%n",
                            timestamp,
                            completedGames,
                            totalRuns,
                            percentage);
                }
                // --------------------------

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}