package be.kuleuven.pylos.battle;

import be.kuleuven.pylos.player.PylosPlayerType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class BattleMT {

    static int N_RUNS_PER_JOB = 2; //has to be even

    public static BattleResult play(PylosPlayerType p1, PylosPlayerType p2, int runs, int nThreads) {
        return play(p1, p2, runs, nThreads, true);
    }

    public static BattleResult play(PylosPlayerType p1, PylosPlayerType p2, int runs, int nThreads, boolean print) {
        int nTasks = runs / N_RUNS_PER_JOB;
        int rest = runs % N_RUNS_PER_JOB;

        ExecutorService pool = Executors.newFixedThreadPool(nThreads);
        List<BattleRunnable> battleRunnables = new ArrayList<>();

        for (int i = 0; i < nTasks; i++) {
            BattleRunnable r = new BattleRunnable(p1, p2, N_RUNS_PER_JOB);
            battleRunnables.add(r);
            pool.execute(r);
        }

        if (rest > 0) {
            BattleRunnable r = new BattleRunnable(p1, p2, rest);
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

        public BattleRunnable(PylosPlayerType p1, PylosPlayerType p2, int nRuns) {
            this.p1 = p1;
            this.p2 = p2;
            this.nRuns = nRuns;
        }

        @Override
        public void run() {
            try {
                result = Battle.play(p1, p2, nRuns, false);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
