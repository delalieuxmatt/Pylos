package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;
import java.util.HashMap;
import java.util.Map;

public class StudentPlayer extends PylosPlayer {
    private PylosGameSimulator simulator;
    private PylosBoard board;
    private double bestMinimax;
    private PylosSphere bestSphere;
    private PylosLocation bestLocation;
    private int branchDepth;
    private final Map<Long, Double> evaluationCache = new HashMap<>();

    private static final double WIN_THIS = 2000;
    private static final double WIN_OTHER = -2000;
    private static final double INITIAL_THIS = -9999;
    private static final double INITIAL_OTHER = 9999;
    private static final int MAX_BRANCH_DEPTH = 6;

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        // make a simulator
        init(game.getState(), board);

        PylosSphere myReserveSphere = board.getReserve(this);
        PylosSphere[] mySpheres = board.getSpheres(this);
        PylosLocation[] locations = board.getLocations();

        // Takes current squares and looks if it could change the position of a ball.
        if (myReserveSphere != null) {
            for (PylosSphere sphere : mySpheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation location : locations) {
                        if (sphere.canMoveTo(location)) {
                            // try's to move the square
                            PylosLocation prevLocation = sphere.getLocation();
                            simulator.moveSphere(sphere, location);
                            double minimax = branchStep(bestMinimax, INITIAL_OTHER);
                            eval(minimax, sphere, location);
                            simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, PLAYER_COLOR);
                        }
                    }
                }
            }
        }

        // Try adding reserve sphere
        for (PylosLocation location : locations) {
            if (location.isUsable()) {
                simulator.moveSphere(myReserveSphere, location);
                double minimax = branchStep(bestMinimax, INITIAL_OTHER);
                eval(minimax, myReserveSphere, location);
                simulator.undoAddSphere(myReserveSphere, PylosGameState.MOVE, PLAYER_COLOR);
            }
        }
        // the best move is saved in bestSphere and bestLocation,
        // need to pass the private args : the function is not inside this method
        game.moveSphere(bestSphere, bestLocation);
    }

    /*
    you have to remove at least one ball when making a square
     */
    public void doRemove(PylosGameIF game, PylosBoard board) {
        // make a simulator
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                //updates the best square to be removed
                tryRemove(sphere, PylosGameState.REMOVE_FIRST);
            }
        }
        // removes this square
        game.removeSphere(bestSphere);
    }

    /***
     * here the computer has the option to remove one square or choose to remove no square
     *
     *
     */
    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                tryRemove(sphere, PylosGameState.REMOVE_SECOND);
            }
        }

        // Try passing
        simulator.pass();
        double minimax = branchStep(bestMinimax, INITIAL_OTHER);
        eval(minimax, null, null);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, PLAYER_COLOR);
        // if there is no best square to remove , do not remove anything
        if (bestSphere != null) {
            game.removeSphere(bestSphere);
        } else {
            game.pass();
        }
    }
    /// code copied from the minimax player from
    private void init(PylosGameState state, PylosBoard board) {
        this.simulator = new PylosGameSimulator(state, PLAYER_COLOR, board);
        this.board = board;
        this.bestMinimax = INITIAL_THIS;
        this.bestSphere = null;
        this.bestLocation = null;
        this.branchDepth = 0;
        this.evaluationCache.clear();
    }
    ///  often used -> one function
    private void eval(double minimax, PylosSphere sphere, PylosLocation location) {
        if (minimax > bestMinimax) {
            bestMinimax = minimax;
            bestSphere = sphere;
            bestLocation = location;
        }
    }
    private void tryRemove(PylosSphere sphere, PylosGameState state) {
        PylosLocation prevLocation = sphere.getLocation();
        simulator.removeSphere(sphere);
        double minimax = branchStep(bestMinimax, INITIAL_OTHER);
        eval(minimax, sphere, null);

        if (state == PylosGameState.REMOVE_FIRST) {
            simulator.undoRemoveFirstSphere(sphere, prevLocation, state, PLAYER_COLOR);
        } else {
            simulator.undoRemoveSecondSphere(sphere, prevLocation, state, PLAYER_COLOR);
        }
    }

    private double branchStep(double alpha, double beta) {
        if (branchDepth == MAX_BRANCH_DEPTH) {
            return evaluatePosition();
        }
        PylosGameState state = simulator.getState();
        branchDepth++;
        double result;

        switch (state) {
            case MOVE:
                result = branchDoMove(alpha, beta);
                break;
            case REMOVE_FIRST:
                result = branchDoRemove(alpha, beta);
                break;
            case REMOVE_SECOND:
                result = branchDoRemoveOrPass(alpha, beta);
                break;
            case COMPLETED:
                result = (simulator.getWinner() == PLAYER_COLOR ? WIN_THIS : WIN_OTHER);
                result += (simulator.getWinner() == PLAYER_COLOR ? -branchDepth : branchDepth);
                break;
            case DRAW:
                result = simulator.getWinner() == PLAYER_COLOR ? WIN_OTHER : WIN_THIS;
                break;
            default:
                throw new IllegalStateException("Game state is: " + state);
        }

        branchDepth--;
        return result;
    }

    private double branchDoMove(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;

        PylosSphere reserve = board.getReserve(color);
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        // Try moving spheres to higher levels
        if (reserve != null) {
            for (PylosSphere sphere : spheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation loc : locations) {
                        if (sphere.canMoveTo(loc)) {
                            PylosLocation prev = sphere.getLocation();
                            simulator.moveSphere(sphere, loc);
                            double result = branchStep(alpha, beta);
                            minimax = updateMinimax(minimax, result, isMaximizing);
                            simulator.undoMoveSphere(sphere, prev, PylosGameState.MOVE, color);
                            if (isMaximizing) {
                                alpha = Math.max(alpha, result);
                            } else {
                                beta = Math.min(beta, result);
                            }
                            if (alpha >= beta) {
                                return minimax;
                            }
                        }
                    }
                }
            }
        }

        // Try adding reserve sphere
        for (PylosLocation loc : locations) {
            if (loc.isUsable()) {
                simulator.moveSphere(reserve, loc);
                double result = branchStep(alpha, beta);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoAddSphere(reserve, PylosGameState.MOVE, color);
                if (isMaximizing) {
                    alpha = Math.max(alpha, result);
                } else {
                    beta = Math.min(beta, result);
                }
                if (alpha >= beta) {
                    return minimax;
                }
            }
        }

        return minimax;
    }

    private double branchDoRemove(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;
        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(alpha, beta);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveFirstSphere(sphere, prev, PylosGameState.REMOVE_FIRST, color);
                if (isMaximizing) {
                    alpha = Math.max(alpha, result);
                } else {
                    beta = Math.min(beta, result);
                }
                if (alpha >= beta) {
                    return minimax;
                }
            }
        }

        return minimax;
    }

    private double branchDoRemoveOrPass(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;

        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(alpha, beta);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveSecondSphere(sphere, prev, PylosGameState.REMOVE_SECOND, color);
                if (isMaximizing) {
                    alpha = Math.max(alpha, result);
                } else {
                    beta = Math.min(beta, result);
                }
                if (alpha >= beta) {
                    return minimax;
                }
            }
        }

        // Pass option
        simulator.pass();
        double result = branchStep(alpha, beta);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, color);
        minimax = updateMinimax(minimax, result, isMaximizing);
        if (isMaximizing) {
            alpha = Math.max(alpha, result);
        } else {
            beta = Math.min(beta, result);
        }
        if (alpha >= beta) {
            return minimax;
        }

        return minimax;
    }

    // Consolidate minimax update logic
    private double updateMinimax(double current, double candidate, boolean isMaximizing) {
        return isMaximizing ? Math.max(current, candidate) : Math.min(current, candidate);
    }

    private double evaluatePosition() {
        long signature = computeBoardSignature();
        Double cachedScore = evaluationCache.get(signature);
        if (cachedScore != null) {
            return cachedScore;
        }

        int myReserves = board.getReservesSize(PLAYER_COLOR);
        int oppReserves = board.getReservesSize(PLAYER_COLOR.other());

        double score = (myReserves - oppReserves) * 100;
        score += evaluateHeightAdvantage() * 50;
        score += evaluateSquarePotential() * 30;
        // evaluation for mobility
        score += (countAvailableMoves(PLAYER_COLOR) - countAvailableMoves(PLAYER_COLOR.other()) )* 10;

        evaluationCache.put(signature, score);
        return score;
    }

    private long computeBoardSignature() {
        long hash = 17;
        for (PylosLocation location : board.getLocations()) {
            hash = hash * 31 + encodeLocation(location);
        }
        hash = hash * 31 + simulator.getState().ordinal();
        hash = hash * 31 + simulator.getColor().ordinal();
        return hash;
    }

    private int encodeLocation(PylosLocation location) {
        PylosSphere sphere = location.getSphere();
        if (sphere == null) {
            return 0;
        }
        return sphere.PLAYER_COLOR == PLAYER_COLOR ? 1 : 2;
    }

    private double evaluateHeightAdvantage() {
        int[] myCounts = new int[4];
        int[] oppCounts = new int[4];

        countHeights(board.getSpheres(PLAYER_COLOR), myCounts);
        countHeights(board.getSpheres(PLAYER_COLOR.other()), oppCounts);

        double score = 0;
        for (int level = 0; level < 4; level++) {
            int weight = 1 << level; // 2^level
            score += (myCounts[level] - oppCounts[level]) * weight;
        }

        return score;
    }

    private void countHeights(PylosSphere[] spheres, int[] counts) {
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null) {
                    counts[loc.Z]++;
                }
            }
        }
    }

    private double evaluateSquarePotential() {
        double score = 0;

        for (int level = 0; level < 3; level++) {
            int size = 4 - level;
            for (int x = 0; x < size - 1; x++) {
                for (int y = 0; y < size - 1; y++) {
                    score += evaluateSingleSquare(x, y, level);
                }
            }
        }

        return score;
    }

    private double evaluateSingleSquare(int x, int y, int level) {
        int myCount = 0, oppCount = 0, emptyCount = 0;
        int[][] offsets = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        for (int[] offset : offsets) {
            PylosLocation loc = board.getBoardLocation(x + offset[0], y + offset[1], level);
            PylosSphere sphere = loc.getSphere();

            if (sphere == null) {
                emptyCount++;
            } else if (sphere.PLAYER_COLOR == PLAYER_COLOR) {
                myCount++;
            } else {
                oppCount++;
            }
        }

        // Only evaluate unmixed squares
        if (oppCount == 0 && myCount == 3) return 5.0;
        if (oppCount == 0 && myCount == 2) return 2.0;
        if (myCount == 0 && oppCount == 3) return -5.0;
        if (myCount == 0 && oppCount == 2) return -2.0;

        return 0.0;
    }
    private int countAvailableMoves(PylosPlayerColor color) {
        int count = 0;
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        // Count moves to higher levels
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                for (PylosLocation loc : locations) {
                    if (sphere.canMoveTo(loc)) {
                        count++;
                    }
                }
            }
        }
        // Count reserve placements
        if (board.getReservesSize(color) > 0) {
            for (PylosLocation loc : locations) {
                if (loc.isUsable()) {
                    count++;
                }
            }
        }
        return count;
    }
}
