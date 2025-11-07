package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.util.*;

/**
 * Created by Jan on 20/02/2015.
 */
public class StudentPlayerRadi extends PylosPlayer {

    public StudentPlayerRadi() {}

    public StudentPlayerRadi(double weightReserves, double weightHeight, double weightCentrality, double weightPotentialSquares) {
        WEIGHT_RESERVES = weightReserves;
        WEIGHT_HEIGHT = weightHeight;
        WEIGHT_CENTRALITY = weightCentrality;
        WEIGHT_POTENTIAL_SQUARES = weightPotentialSquares;
    }
    // Config
    private static final int MAX_DEPTH = 9;

    // Evaluation weights
    private static double WEIGHT_RESERVES = 23;
    private static double WEIGHT_HEIGHT = 10;
    private static double WEIGHT_CENTRALITY = 2;
    private static double WEIGHT_POTENTIAL_SQUARES = 18;

    // Late Move Reduction parameters
    final double R_BASE = 1.0;
    final double R_DEPTH_SCALE = 2;

    // Game State
    private PylosBoard board;
    private PylosGameSimulator simulator;
    private PylosPlayerColor myColor;

    // Best Move Tracking
    private PylosSphere bestSphere;

    private boolean RANDOM = false;

    // Helper class for cache entry
    private static class CacheEntry {
        double score;
        int depth;

        CacheEntry(double score, int depth) {
            this.score = score;
            this.depth = depth;
        }
    }

    // Helper class for moves
    private static class Move {
        PylosSphere sphere;
        PylosLocation location;

        Move(PylosSphere sphere, PylosLocation location) {
            this.sphere = sphere;
            this.location = location;
        }
    }

    // Helper class for moves that have received a score
    static class ScoredMove {
        Move move;
        double score;

        ScoredMove(Move move, double score) {
            this.move = move;
            this.score = score;
        }
    }


    private ArrayList<Move> sortedMoves() {
        ArrayList<Move> priority1 = new ArrayList<>(); // move a sphere to complete a square
        ArrayList<Move> priority2 = new ArrayList<>(); // move a sphere to a higher level
        ArrayList<Move> priority3 = new ArrayList<>(); // place a reserve sphere in a square
        ArrayList<Move> priority4 = new ArrayList<>(); // Block opponent's square
        ArrayList<Move> priority5 = new ArrayList<>(); // Other moves

        PylosSphere myReserve = board.getReserve(myColor);
        PylosSphere[] mySpheres = board.getSpheres(myColor);
        PylosLocation[] locations = board.getLocations();

        // check for higher priority moves first
        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canMove()) {
                for (PylosLocation location : locations) {
                    if (sphere.canMoveTo(location) && location.isUsable()) {
                        boolean completesSquare = false;
                        boolean blocksOpponent = false;
                        for (PylosSquare square : location.getSquares()) {
                            if (square.getInSquare(myColor) == 3 && !location.isUsed()) {
                                completesSquare = true;
                                break;
                            }
                            if (square.getInSquare(myColor.other()) == 3 && !location.isUsed()) {
                                blocksOpponent = true;
                            }
                        }

                        if (completesSquare) {
                            priority1.add(new Move(sphere, location));
                        } else if (location.Z > sphere.getLocation().Z) {
                            priority2.add(new Move(sphere, location));
                        } else if (blocksOpponent) {
                            priority4.add(new Move(sphere, location));
                        } else {
                            priority5.add(new Move(sphere, location));
                        }

                    }
                }
            }
        }

        // try placing a reverse sphere
        if (myReserve != null) {
            for (PylosLocation location : locations) {
                if (myReserve.canMoveTo(location) && location.isUsable()) {
                    boolean completesSquare = false;
                    boolean blocksOpponent = false;
                    for (PylosSquare square : location.getSquares()) {
                        if (square.getInSquare(myColor) == 3 && !location.isUsed()) {
                            completesSquare = true;
                            break;
                        }
                        if (square.getInSquare(myColor.other()) == 3 && !location.isUsed()) {
                            blocksOpponent = true;
                        }
                    }
                    if (completesSquare) {
                        priority3.add(new Move(myReserve, location));
                    } else if (blocksOpponent) {
                        priority4.add(new Move(myReserve, location));
                    } else {
                        priority5.add(new Move(myReserve, location));
                    }
                }
            }
        }

        // Combine all moves in order
        ArrayList<Move> allMoves = new ArrayList<>();
        allMoves.addAll(priority1);
        allMoves.addAll(priority2);
        allMoves.addAll(priority3);
        allMoves.addAll(priority4);
        allMoves.addAll(priority5);

        return allMoves;
    }


    // Use a HashMap for caching
    private final HashMap<String, CacheEntry> transpositionTable = new HashMap<>();

    // cache the board, the player color of the one doing the move and the game state (move, remove, removeSecond)
    private String getCacheKey(PylosBoard board, PylosPlayerColor color, PylosGameState state) {
        return board.toLong() + "_" + color.ordinal() + "_" + state.ordinal();
    }


    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(board, game.getState());

        ArrayList<Move> allMoves = sortedMoves();

        double bestValue = Double.NEGATIVE_INFINITY;
        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;
        Move bestMove = null;
        ArrayList<Move> bestMoves = new ArrayList<>();
        ArrayList<ScoredMove> scoredMoves = new ArrayList<>();


        for (int i = 0; i < allMoves.size(); i++) {
            Move move = allMoves.get(i);
            double value;
            int reducedDepth = MAX_DEPTH;

            // Store original location before the move
            PylosLocation originalLocation = move.sphere.isReserve() ? null : move.sphere.getLocation();

            simulator.moveSphere(move.sphere, move.location);

            // Check for symmetric variants in cache
            boolean variantInCache = isVariant(board.toLong(), simulator.getColor(), simulator);

            if (variantInCache) {
                // Undo move using the stored original location
                if (originalLocation == null) { // reserve sphere was placed
                    simulator.undoAddSphere(move.sphere, PylosGameState.MOVE, this.PLAYER_COLOR);
                } else { // moved sphere
                    simulator.undoMoveSphere(move.sphere, originalLocation, PylosGameState.MOVE, this.PLAYER_COLOR);
                }

                continue; // skip this move as its variant is already in cache
                // every time we enter doMove(), we clear the cache table to avoid conflicts with earlier moves
            }

            // Apply Late Move Reduction, chose 4 as the threshold
            if (i < 4) {
                value = minimax(reducedDepth - 1, alpha, beta);
            } else {
                double reduction = R_BASE + (i / R_DEPTH_SCALE);
                reducedDepth = Math.max(1, (int) Math.round(MAX_DEPTH - reduction));
                value = minimax(reducedDepth - 1, alpha, beta);

                if (value >= alpha && reducedDepth < MAX_DEPTH) {
                    // Re-search at full depth if the move looks promising
                    reducedDepth = MAX_DEPTH;
                    value = minimax(reducedDepth - 1, alpha, beta);
                }
            }

            // add every scored move for potential random selection later
            scoredMoves.add(new ScoredMove(move, value));

            // Undo move using the stored original location
            if (originalLocation == null) {
                simulator.undoAddSphere(move.sphere, PylosGameState.MOVE, this.PLAYER_COLOR);
            } else {
                simulator.undoMoveSphere(move.sphere, originalLocation, PylosGameState.MOVE, this.PLAYER_COLOR);
            }

            // Update best move if necessary
            if (value > bestValue) {
                bestMove = move;
                bestValue = value;
            }

            // Alpha-beta pruning
            if (bestValue > alpha) alpha = bestValue;
            if (beta <= alpha) break; // beta cut-off

        }

        if (bestMove != null) {
            if (RANDOM) {
                // Sort moves by score in descending order
                scoredMoves.sort((a, b) -> Double.compare(b.score, a.score));

                // Find all moves with the best score
                double bestScore = scoredMoves.get(0).score;

                for (ScoredMove sm : scoredMoves) {
                    if (sm.score == bestScore) {
                        bestMoves.add(sm.move);
                    } else {
                        break; // Since sorted, we can stop when score differs
                    }
                }

                Collections.shuffle(bestMoves, getRandom());
                bestMove = bestMoves.get(0);

                // Randomly select from best moves
                game.moveSphere(bestMove.sphere, bestMove.location);
            } else {
                // Deterministically select the best move
                game.moveSphere(bestMove.sphere, bestMove.location);
            }
        } else {
            throw new IllegalStateException("No valid move found");
        }
    }

    private ArrayList<PylosSphere> sortedRemoveMoves() {
        ArrayList<PylosSphere> priority = new ArrayList<>(); // move a sphere to complete a square
        PylosSphere[] mySpheres = board.getSpheres(myColor);

        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canRemove()) {
                priority.add((sphere));
            }
        }

        // sort by height (Z coordinate), removing lower spheres first
        priority.sort(Comparator.comparingInt(a -> a.getLocation().Z));
        return priority;
    }


    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        init(board, game.getState());
        double bestValue = Double.NEGATIVE_INFINITY;
        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;

        ArrayList<PylosSphere> sortedSpheres = sortedRemoveMoves();
        ArrayList<ScoredMove> scoredMoves = new ArrayList<>();
        ArrayList<PylosSphere> bestSpheres = new ArrayList<>();

        for (int i = 0; i < sortedSpheres.size(); i++) {
            double value;
            PylosSphere sphere = sortedSpheres.get(i);
            PylosLocation prevLocation = sphere.getLocation();
            simulator.removeSphere(sphere);
            int reducedDepth = MAX_DEPTH;

            // LMR
            if (i < 4) {
                value = minimax(reducedDepth - 1, alpha, beta);
            } else {
                double reduction = R_BASE + (i / R_DEPTH_SCALE);
                reducedDepth = Math.max(1, (int) Math.round(MAX_DEPTH - reduction));

                value = minimax(reducedDepth - 1, alpha, beta);

                if (value >= alpha && reducedDepth < MAX_DEPTH) {
                    // Re-search at full depth if the move looks promising
                    reducedDepth = MAX_DEPTH;
                    value = minimax(reducedDepth - 1, alpha, beta);
                }
            }

            scoredMoves.add(new ScoredMove(new Move(sphere, null), value));

            if (value > bestValue) {
                bestValue = value;
                bestSphere = sphere;
            }

            if (bestValue > alpha) alpha = bestValue;
            simulator.undoRemoveFirstSphere(sphere, prevLocation, PylosGameState.REMOVE_FIRST, this.PLAYER_COLOR);
            if (beta <= alpha) break; // beta cut-off
        }

        if (bestSphere != null) {
            if (RANDOM) {
                // Sort moves by score in descending order
                scoredMoves.sort((a, b) -> Double.compare(b.score, a.score));

                // Find all moves with the best score
                double bestScore = scoredMoves.get(0).score;

                for (ScoredMove sm : scoredMoves) {
                    if (sm.score == bestScore) {
                        bestSpheres.add(sm.move.sphere);
                    } else {
                        break; // Since sorted, we can stop when score differs
                    }
                }

                Collections.shuffle(bestSpheres, getRandom());
                bestSphere = bestSpheres.get(0);

                // Randomly select from best moves
                game.removeSphere(bestSphere);
            } else {
                // Deterministically select the best move
                game.removeSphere(bestSphere);
            }

        } else {
            throw new IllegalStateException("No valid removal found");
        }
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        init(board, game.getState());
        double bestValue = Double.NEGATIVE_INFINITY;
        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;
        boolean shouldPass = false;

        ArrayList<PylosSphere> sortedSpheres = sortedRemoveMoves();
        ArrayList<ScoredMove> scoredMoves = new ArrayList<>();
        ArrayList<PylosSphere> bestSpheres = new ArrayList<>();

        for (int i = 0; i < sortedSpheres.size(); i++) {
            double value;
            PylosSphere sphere = sortedSpheres.get(i);
            PylosLocation prevLocation = sphere.getLocation();
            simulator.removeSphere(sphere);

            // LMR
            if (i < 4) {
                value = minimax(MAX_DEPTH - 1, alpha, beta);
            } else {
                double reduction = R_BASE + (i / R_DEPTH_SCALE);
                int reducedDepth = Math.max(1, (int) Math.round(MAX_DEPTH - reduction));

                value = minimax(reducedDepth - 1, alpha, beta);

                if (value >= alpha && reducedDepth < MAX_DEPTH) {
                    // Re-search at full depth if the move looks promising
                    value = minimax(MAX_DEPTH - 1, alpha, beta);
                }
            }

            scoredMoves.add(new ScoredMove(new Move(sphere, null), value));

            if (value > bestValue) {
                bestValue = value;
                bestSphere = sphere;
            }
            if (bestValue > alpha) alpha = bestValue;
            simulator.undoRemoveSecondSphere(sphere, prevLocation, PylosGameState.REMOVE_SECOND, this.PLAYER_COLOR);
            if (beta <= alpha) break; // beta cut-off
        }

        // check passing
        simulator.pass();

        double value = minimax(MAX_DEPTH - 1, alpha, beta);

        if (value > bestValue) {
            bestValue = value;
            bestSphere = null;
            shouldPass = true;
        }

        if (bestValue > alpha) alpha = bestValue;

        simulator.undoPass(PylosGameState.REMOVE_SECOND, this.PLAYER_COLOR);

        if (shouldPass) {
            game.pass();
        } else {
            if (RANDOM) {
                // Sort moves by score in descending order
                scoredMoves.sort((a, b) -> Double.compare(b.score, a.score));

                // Find all moves with the best score
                double bestScore = scoredMoves.get(0).score;

                for (ScoredMove sm : scoredMoves) {
                    if (sm.score == bestScore) {
                        bestSpheres.add(sm.move.sphere);
                    } else {
                        break; // Since sorted, we can stop when score differs
                    }
                }

                Collections.shuffle(bestSpheres, getRandom());
                bestSphere = bestSpheres.get(0);

                // Randomly select from best moves
                game.removeSphere(bestSphere);
            } else {
                game.removeSphere(bestSphere);
            }
        }
    }

    private void init(PylosBoard board, PylosGameState state) {
        this.board = board;
        this.myColor = this.PLAYER_COLOR;
        this.simulator = new PylosGameSimulator(state, myColor, board);
        this.bestSphere = null;
        transpositionTable.clear(); // clear cache for new move
    }

    /* Minimax algorithm */
    private double minimax(int depth, double alpha, double beta) {
        PylosPlayerColor currentPlayer = simulator.getColor();
        double result = currentPlayer == myColor ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        if (depth == 0 || simulator.getState() == PylosGameState.COMPLETED || simulator.getState() == PylosGameState.DRAW) {
            return evaluate();
        }

        //  cashing logic
        String key = getCacheKey(board, simulator.getColor(), simulator.getState());
        CacheEntry entry = transpositionTable.get(key);
        if (entry != null && entry.depth >= depth) {
            return entry.score;
        }

        PylosGameState state = simulator.getState();

        switch (state) {
            case MOVE:
                result = evalAllMoves(depth, currentPlayer, alpha, beta);
                assert simulator.getColor() == currentPlayer && simulator.getState() == state;
                transpositionTable.put(key, new CacheEntry(result, depth));
                return result;
            case REMOVE_FIRST:
                result = evalAllRemovals(depth, currentPlayer, alpha, beta);
                assert simulator.getColor() == currentPlayer && simulator.getState() == state;
                transpositionTable.put(key, new CacheEntry(result, depth));
                return result;
            case REMOVE_SECOND:
                result = evalAllRemovalsOrPass(depth, currentPlayer, alpha, beta);
                assert simulator.getColor() == currentPlayer && simulator.getState() == state;
                transpositionTable.put(key, new CacheEntry(result, depth));
                return result;
        }

        return result;
    }

    private double evalAllMoves(int depth, PylosPlayerColor currentPlayer, double alpha, double beta) {
        double bestValue = currentPlayer == myColor ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        PylosSphere myReserve = board.getReserve(currentPlayer);
        PylosSphere[] mySpheres = board.getSpheres(currentPlayer);
        PylosLocation[] locations = board.getLocations();

        // try placing a sphere to a highest possible location
        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canMove()) {
                for (PylosLocation location : locations) {
                    if (sphere.canMoveTo(location) && location.isUsable()) {
                        PylosLocation prevLocation = sphere.getLocation();
                        simulator.moveSphere(sphere, location);

                        double value = minimax(depth - 1, alpha, beta);

                        if (currentPlayer == PLAYER_COLOR) {
                            if (value > bestValue) bestValue = value;
                            if (bestValue > alpha) alpha = bestValue;
                        } else {
                            if (value < bestValue) bestValue = value;
                            if (bestValue < beta) beta = bestValue;
                        }

                        simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, currentPlayer);
                        assert simulator.getState() == PylosGameState.MOVE && simulator.getColor() == currentPlayer : simulator.getState() + " " + simulator.getColor() + "\tshould be: " + PylosGameState.MOVE + " " + currentPlayer;
                        if (beta <= alpha) break;

                    }
                }
            }
        }

        // try placing a reserve sphere to a highest possible location
        if (myReserve != null) {
            for (PylosLocation location : locations) {
                if (myReserve.canMoveTo(location) && location.isUsable()) {
                    simulator.moveSphere(myReserve, location);

                    double value = minimax(depth - 1, alpha, beta);

                    if (currentPlayer == PLAYER_COLOR) {
                        if (value > bestValue) bestValue = value;
                        if (bestValue > alpha) alpha = bestValue;
                    } else {
                        if (value < bestValue) bestValue = value;
                        if (bestValue < beta) beta = bestValue;
                    }

                    simulator.undoAddSphere(myReserve, PylosGameState.MOVE, currentPlayer);
                    assert simulator.getState() == PylosGameState.MOVE && simulator.getColor() == currentPlayer : simulator.getState() + " " + simulator.getColor() + "\tshould be: " + PylosGameState.MOVE + " " + currentPlayer;

                    if (beta <= alpha) break;
                }
            }
        }

        assert !(currentPlayer == PLAYER_COLOR && bestValue == Double.NEGATIVE_INFINITY);
        assert !(currentPlayer != PLAYER_COLOR && bestValue == Double.POSITIVE_INFINITY);
        return bestValue;
    }

    private double evalAllRemovals(int depth, PylosPlayerColor currentPlayer, double alpha, double beta) {
        double bestValue = currentPlayer == myColor ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        PylosSphere[] mySpheres = board.getSpheres(currentPlayer);

        // try removing each sphere
        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);

                double value = minimax(depth - 1, alpha, beta);

                if (currentPlayer == PLAYER_COLOR) {
                    if (value > bestValue) bestValue = value;
                    if (bestValue > alpha) alpha = bestValue;
                } else {
                    if (value < bestValue) bestValue = value;
                    if (bestValue < beta) beta = bestValue;
                }

                simulator.undoRemoveFirstSphere(sphere, prevLocation, PylosGameState.REMOVE_FIRST, currentPlayer);
                assert simulator.getState() == PylosGameState.REMOVE_FIRST && simulator.getColor() == currentPlayer : simulator.getState() + " " + simulator.getColor() + "\tshould be: " + PylosGameState.REMOVE_FIRST + " " + currentPlayer;

                if (beta <= alpha) break;
            }
        }

        assert !(currentPlayer == PLAYER_COLOR && bestValue == Double.NEGATIVE_INFINITY);
        assert !(currentPlayer != PLAYER_COLOR && bestValue == Double.POSITIVE_INFINITY);
        return bestValue;
    }

    private double evalAllRemovalsOrPass(int depth, PylosPlayerColor currentPlayer, double alpha, double beta) {
        double bestValue = currentPlayer == myColor ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        PylosSphere[] mySpheres = board.getSpheres(currentPlayer);


        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);

                double value = minimax(depth - 1, alpha, beta);

                if (currentPlayer == PLAYER_COLOR) {
                    if (value > bestValue) bestValue = value;
                    if (bestValue > alpha) alpha = bestValue;
                } else {
                    if (value < bestValue) bestValue = value;
                    if (bestValue < beta) beta = bestValue;
                }

                simulator.undoRemoveSecondSphere(sphere, prevLocation, PylosGameState.REMOVE_SECOND, currentPlayer);
                assert simulator.getState() == PylosGameState.REMOVE_SECOND && simulator.getColor() == currentPlayer : simulator.getState() + " " + simulator.getColor() + "\tshould be: " + PylosGameState.REMOVE_SECOND + " " + currentPlayer;

                if (beta <= alpha) break;
            }
        }

        // consider passing
        simulator.pass();

        double value = minimax(depth - 1, alpha, beta);

        if (currentPlayer == PLAYER_COLOR) {
            if (value > bestValue) bestValue = value;
            if (bestValue > alpha) alpha = bestValue;
        } else {
            if (value < bestValue) bestValue = value;
            if (bestValue < beta) beta = bestValue;
        }

        simulator.undoPass(PylosGameState.REMOVE_SECOND, currentPlayer);
        assert simulator.getState() == PylosGameState.REMOVE_SECOND && simulator.getColor() == currentPlayer : simulator.getState() + " " + simulator.getColor() + "\tshould be: " + PylosGameState.REMOVE_SECOND + " " + currentPlayer;
        if (beta <= alpha) return bestValue;

        assert !(currentPlayer == PLAYER_COLOR && bestValue == Double.NEGATIVE_INFINITY);
        assert !(currentPlayer != PLAYER_COLOR && bestValue == Double.POSITIVE_INFINITY);
        return bestValue;
    }

    private double evaluate() {
        if (simulator.getState() == PylosGameState.COMPLETED) {
            return simulator.getWinner() == myColor ? 10000 : -10000;
        }

        if (simulator.getState() == PylosGameState.DRAW) {
            return 0;
        }
        PylosPlayerColor oppColor = myColor.other();

        // Factor 1: Number of reserves
        double myReserves = board.getReservesSize(PLAYER_COLOR);
        double oppReserves = board.getReservesSize(oppColor);

        // Factor 2: Height of highest sphere and Factor 3: Centrality of spheres
        double myCentrality = 0, oppCentrality = 0;
        int myMaxHeight = 0, oppMaxHeight = 0;
        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                int h = loc.Z;
                if (h > myMaxHeight) myMaxHeight = h;
                double distToCenter = Math.abs(loc.X - 1.5) + Math.abs(loc.Y - 1.5);
                myCentrality += (3.0 - distToCenter);
            }
        }

        for (PylosSphere sphere : board.getSpheres(oppColor)) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                int h = loc.Z;
                if (h > oppMaxHeight) oppMaxHeight = h;
                double distToCenter = Math.abs(loc.X - 1.5) + Math.abs(loc.Y - 1.5);
                oppCentrality += (3.0 - distToCenter);
            }
        }

        // Factor 4: Potential squares
        double myPotentialSquares = 0, oppPotentialSquares = 0;
        for (PylosSquare sq : board.getAllSquares()) {
            int myCount = sq.getInSquare(myColor);
            int oppCount = sq.getInSquare(oppColor);
            int empty = 4 - (myCount + oppCount);

            if (myCount == 3 && empty == 1) {
                myPotentialSquares++;
            } else if (oppCount == 3 && empty == 1) {
                oppPotentialSquares++;
            }
        }

        double reserveDiff = myReserves - oppReserves;
        double heightDiff = myMaxHeight - oppMaxHeight;
        double centralityDiff = myCentrality - oppCentrality;
        double potentialSquaresDiff = myPotentialSquares - oppPotentialSquares;

        return (WEIGHT_RESERVES * reserveDiff +
                WEIGHT_HEIGHT * heightDiff +
                WEIGHT_CENTRALITY * centralityDiff +
                WEIGHT_POTENTIAL_SQUARES * potentialSquaresDiff);
    }

    public boolean isVariant(long state, PylosPlayerColor color, PylosGameSimulator simulator) {
        long variantX = flipOverX(state);
        String keyPotentiallyInCacheVarX = variantX + "_" + color.ordinal() + simulator.getState().ordinal();
        if (transpositionTable.containsKey(keyPotentiallyInCacheVarX)) {
            //return variantX;
            return true;
        }

        long variantY = flipOverY(state);
        String keyPotentiallyInCacheVarY = variantY + "_" + color.ordinal() + simulator.getState().ordinal();
        if (transpositionTable.containsKey(keyPotentiallyInCacheVarY)) {
            //return variantY;
            return true;
        }

        long variant90 = rotateRight_90(state);
        String keyPotentiallyInCacheVar90 = variant90 + "_" + color.ordinal() + simulator.getState().ordinal();
        if (transpositionTable.containsKey(keyPotentiallyInCacheVar90)) {
            //return variant90;
            return true;
        }

        long variant180 = rotateRight_90(variant90);
        String keyPotentiallyInCacheVar180 = variant180 + "_" + color.ordinal() + simulator.getState().ordinal();
        if (transpositionTable.containsKey(keyPotentiallyInCacheVar180)) {
            //return variant180;
            return true;
        }

        long variant270 = rotateRight_90(variant180);
        String keyPotentiallyInCacheVar270 = variant270 + "_" + color.ordinal() + simulator.getState().ordinal();
        if (transpositionTable.containsKey(keyPotentiallyInCacheVar270)) {
            //return variant270;
            return true;
        }

        //return 0xFFFFFFFFFFFFFFFFL; //kunnen we nooit hebben
        return false;
    }

    public static long flipOverX(long state) {
        long result = 0L;

        //layer one (z=0):
        /* x=3      x=2      x=1      x=0
         * xxxxxx01 xxxxxxxx xxxxxxxx xxxxxxxx
         * p3       p2       p1       p0
         * => light sphere op (x=3, y=0, z=0) wanneer we deze spiegelen rond de x-as krijgen we (x=0, y=0, z=0)
         */
        result |= ((state & 0xFF000000L) >>> 24); //l1_p3_to_p0
        result |= (state & 0xFF0000L) >>> 8; //l1_p2_to_p1
        result |= (state & 0xFF00L) << 8; //l1_p1_to_p2
        result |= (state & 0xFFL) << 24; //l1_p0_to_p3

        //layer two (z=1):
        /* x=2    x=1    x=0
         * xxxx01 xxxxxx xxxxxx
         * p2     p1     p0
         * => light sphere op (x=2, y=0, z=1) wanneer we deze spiegelen rond de x-as krijgen we (x=0, y=0, z=1)
         */
        if ((state & 0x3FFFF00000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer twee liggen
            result |= (state & 0x3F00000000000L) >>> 12; //l2_p2_to_p0
            result |= (state & 0xFC000000000L); //l2_p1 --> wordt niet gedraaid (deze spheres zitten/liggen OP de symmetrie-as)
            result |= (state & 0x3F00000000L) << 12; //l2_p0_to_p2
        } else {
            return result;
        }

        //layer three (z=2):
        /* x=1  x=0
         * xx01 xxxx
         * p1   p0
         * => light sphere op (x=1, y=0, z=2) wanneer we deze spiegelen rond de x-as krijgen we (x=0, y=0, z=2)
         */
        if ((state & 0x3FC000000000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer drie liggen
            result |= (state & 0x3C0000000000000L) >>> 4; //l3_p1_to_p0
            result |= (state & 0x3C000000000000L) << 4; //l2_p0_to_p1
        } else {
            return result;
        }

        //layer four (z=3):
        //er moet niks gespiegeld worden want de sphere ligt op de symmetrie as
        if ((state & 0xC00000000000000L) != 0L) {
            result |= (state & 0xC00000000000000L);
        } else {
            return result;
        }

        return result;
    }

    private static long flipOverY(long state) {
        long result = 0L;

        //layer one (z=0):
        /* x=3      x=2      x=1       x=0
         *                             y=3 y=2 y=1 y=0
         * xxxxxxxx xxxxxxxx xxxxxxxx [xx  xx  xx  01]
         *                             p3  p2  p1  p0
         * 11000000 11000000 11000000  11  00  00  00 => 0xC0C0C0C0L
         * 00110000 00110000 00110000  00  11  00  00 => 0x30303030L
         * ...
         * => light sphere op (x=0, y=0, z=0) wanneer we deze spiegelen rond de y-as krijgen we (x=0, y=3, z=0)
         */
        result |= (state & 0xC0C0C0C0L) >>> 6; //l1_x0_p3_to_p0, l1_x1_p3_to_p0, l1_x2_p3_to_p0, l1_x3_p3_to_p0
        result |= (state & 0x30303030L) >>> 2; //l1_x0_p2_to_p1, l1_x1_p2_to_p1, l1_x2_p2_to_p1, l1_x3_p2_to_p1
        result |= (state & 0xC0C0C0CL) << 2; //l1_x0_p1_to_p2, l1_x1_p1_to_p2, l1_x2_p1_to_p2, l1_x3_p1_to_p2
        result |= (state & 0x3030303L) << 6; //l1_x0_p0_to_p3, l1_x1_p0_to_p3, l1_x2_p0_to_p3, l1_x3_p0_to_p3

        //layer two (z=1):
        /* x=2    x=1     x=0
         *                y=2 y=1 y=0
         * xxxxxx xxxxxx [xx  xx  01]
         *                p2  p1  p0
         * 110000 110000  11  00  00 => 0x30C3000000000L
         * 001100 001100  00  11  00 => 0x1830C00000000L
         * ...
         * => light sphere op (x=0, y=0, z=1) wanneer we deze spiegelen rond de y-as krijgen we (x=0, y=2, z=1)
         */
        if ((state & 0x3FFFF00000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer twee liggen
            result |= (state & 0x30C3000000000L) >>> 4; //l2_x0_p2_to_p0, l2_x1_p2_to_p0, l2_x2_p2_to_p0
            result |= (state & 0x1830C00000000L); //l2_x0_p1, l2_x1_p1, l2_x2_p1 --> wordt niet gedraaid (deze spheres zitten/liggen OP de symmetrie-as)
            result |= (state & 0x30C300000000L) << 4; //l2_x0_p0_to_p2, l2_x1_p0_to_p2, l2_x2_p0_to_p2
        } else {
            return result;
        }

        //layer three (z=2):
        /* x=1   x=0
         *       y=1 y=0
         * xxxx [xx  x1]
         *       p1  p0
         * 1100  11  00 => 0x330000000000000L
         * 0011  00  11 => 0xCC000000000000L
         * => light sphere op (x=0, y=0, z=2) wanneer we deze spiegelen rond de y-as krijgen we (x=0, y=1, z=2)
         */
        if ((state & 0x3FC000000000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer drie liggen
            result |= (state & 0x330000000000000L) >>> 2; //l3_x0_p1_to_p0, l3_x1_p1_to_p0
            result |= (state & 0xCC000000000000L) << 2; //l3_x0_p0_to_p1, l3_x1_p0_to_p1
        } else {
            return result;
        }

        //layer four (z=3):
        //er moet niks gespiegeld worden want de sphere ligt op de symmetrie as
        if ((state & 0xC00000000000000L) != 0L) {
            result |= (state & 0xC00000000000000L);
        } else {
            return result;
        }

        return result;
    }

    private static long rotateRight_90(long state) {
        long result = 0L;

        //layer one (z=0):
        //column 0 to row 0:
        /*
         *  column 0
         *  |
         * [0|0| | ]    [x|0|x|0] -- row 0
         * [x|x| | ] => [ | |x|0]
         * [0| |0| ]    [ |0| | ]
         * [x| | |x]    [x| | | ]
         * state 1   => state2
         * state 1: 0000000000000000000000000000000000000000000000000000000001100110
         * state 2: 0000000000000000000000000000000001000000100000000100000010000000
         */
        result |= (state & 0x3L) << 6; //l1_x0y0_to_x0y3
        result |= (state & 0xCL) << 12; //l1_x0y1_to_x1y3
        result |= (state & 0x30L) << 18; //l1_x0y2_to_x2y3
        result |= (state & 0xC0L) << 24; //l1_x0y3_to_x3y3

        //column 1 to row 1:
        result |= (state & 0x300L) >>> 4; //l1_x1y0_to_x0y2
        result |= (state & 0xC00L) << 2; //l1_x1y1_to_x1y2
        result |= (state & 0x3000L) << 8; //l1_x1y2_to_x2y2
        result |= (state & 0xC000L) << 14; //l1_x1y3_to_x3y2

        //column 2 to row 2:
        result |= (state & 0x30000L) >>> 14; //l1_x2y0_to_x0y1
        result |= (state & 0xC0000L) >>> 8; //l1_x2y1_to_x1y1
        result |= (state & 0x300000L) >>> 2; //l1_x2y2_to_x2y1
        result |= (state & 0xC00000L) << 4; //l1_x2y3_to_x3y1

        //column 3 to row 3:
        result |= (state & 0x3000000L) >>> 24; //l1_x3y0_to_x0y0
        result |= (state & 0xC000000L) >>> 18; //l1_x3y1_to_x1y0
        result |= (state & 0x30000000L) >>> 12; //l1_x3y2_to_x2y0
        result |= (state & 0xC0000000L) >>> 6; //l1_x3y3_to_x3y0

        if ((state & 0x3FFFF00000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer twee liggen
            //layer two (z=1):
            //column 0 to row 0:
            /*
             *  column 0
             *  |
             * [0|0| ]    [0|x|0] -- row 0
             * [x|*| ] => [ |*|0]
             * [0| |0]    [0| | ]
             * state 1 => state2
             */
            result |= (state & 0x300000000L) << 4; //l2_x0y0_to_x0y2
            result |= (state & 0xC00000000L) << 8; //l2_x0y1_to_x1y2
            result |= (state & 0x3000000000L) << 12; //l2_x0y2_to_x2y2

            //column 1 to row 1:
            result |= (state & 0xC000000000L) >>> 4; //l2_x1y0_to_x0y1
            result |= (state & 0x30000000000L); //l2_x1y1_to_x1y1 //deze sphere moet niet verplaatst worden (zie *)
            result |= (state & 0xC0000000000L) << 4; //l2_x1y2_to_x2y1

            //column 2 to row 2:
            result |= (state & 0x300000000000L) >>> 12; //l2_x2y0_to_x0y0
            result |= (state & 0xC00000000000L) >>> 8; //l2_x2y1_to_x1y0
            result |= (state & 0x3000000000000L) >>> 4; //l2_x2y2_to_x2y0
        } else {
            return result;
        }

        if ((state & 0x3FC000000000000L) != 0L) { //nagaan of er uberhaupt wel spheres op layer drie liggen
            //layer three (z=2):
            //column 0 to row 0:
            /*
             *  column 0
             *  |
             * [0|0]      [x|0] -- row 0
             * [x|x]   => [x|0]
             * state 1 => state2
             */
            result |= (state & 0xC000000000000L) << 2; //l3_x0y0_to_x0y1
            result |= (state & 0x30000000000000L) << 4; //l3_x0y1_to_x1y1

            //column 1 to row 1:
            result |= (state & 0xC0000000000000L) >>> 4; //l3_x1y0_to_x0y0
            result |= (state & 0x300000000000000L) >>> 2; //l3_x1y1_to_x1y0
        } else {
            return result;
        }

        //layer four (z=3):
        //er moet niks geroteerd worden want deze sphere moet niet verplaatst worden
        if ((state & 0xC00000000000000L) != 0L) {
            result |= (state & 0xC00000000000000L);
        } else {
            return result;
        }

        return result;
    }
}
