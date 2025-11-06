package be.kuleuven.pylos;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PylosPlayerML extends PylosPlayer {

    private final SavedModelBundle model;

    /**
     * NEW: Set the search depth for the MiniMax algorithm.
     * 3-4 is a good starting point.
     * Your original code was equivalent to SEARCH_DEPTH = 1.
     */
    private static final int SEARCH_DEPTH = 4;

    private Map<Long, Float> transpositionTable;


    public PylosPlayerML(SavedModelBundle model) {
        this.model = model;
        this.transpositionTable = new HashMap<>();
    }

    /**
     * MODIFIED: Calls the new findBestAction method with the search depth.
     */
    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.MOVE, SEARCH_DEPTH);
        bestAction.execute(game);
    }

    /**
     * MODIFIED: Calls the new findBestAction method with the search depth.
     */
    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_FIRST, SEARCH_DEPTH);
        bestAction.execute(game);
    }

    /**
     * MODIFIED: Calls the new findBestAction method with the search depth.
     */
    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_SECOND, SEARCH_DEPTH);
        bestAction.execute(game);
    }

    /**
     * NEW: This is the "root" of the NegaMax search.
     * It iterates through all possible moves (1-ply) and calls the recursive
     * negamax() function to evaluate the resulting board state.
     *
     * @param board The current board
     * @param color The player whose turn it is
     * @param state The current game state
     * @param depth The max depth to search
     * @return The best Action
     */
    private Action findBestAction(PylosBoard board, PylosPlayerColor color, PylosGameState state, int depth) {
        this.transpositionTable.clear();
        List<Action> actionList = new ArrayList<>();
        List<Action> actions = generateActions(board, color, state, actionList);
        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);

        Action bestAction = null;
        float bestEval = Float.NEGATIVE_INFINITY;

        for (Action action : actions) {
            action.simulate(simulator);

            // Call the recursive negamax function for the *opponent*
            // The score returned is from the opponent's perspective, so we negate it.
            float eval = -negamax(board, simulator.getColor(), simulator.getState(), depth - 1, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);

            action.reverseSimulate(simulator);

            if (eval > bestEval) {
                bestEval = eval;
                bestAction = action;
            }
        }

        // Failsafe: if no best action is found (e.g., all moves lead to a loss)
        // just pick the first one.
        if (bestAction == null && !actions.isEmpty()) {
            return actions.get(0);
        }

        return bestAction;
    }

    /**
     * NEW: The recursive NegaMax search function.
     * This function returns a *score*, not an Action.
     *
     * @param board   The board (mutated by the simulator)
     * @param color   The *current* player in this simulation
     * @param state   The *current* state in this simulation
     * @param depth   The remaining depth to search
     * @return The score of this board from the perspective of the *current* player (color)
     */
    private float negamax(PylosBoard board, PylosPlayerColor color, PylosGameState state, int depth, float alpha, float beta) {

        // Base Case 1: Terminal Node (Game is over)
        // Check if the game is completed
        long boardKey = board.toLong();
        if (transpositionTable.containsKey(boardKey)) {
            return transpositionTable.get(boardKey);
        }
        if (state == PylosGameState.COMPLETED) {
            // 'state' is COMPLETED, which means the *previous* move was a winning move.
            // The player whose turn it is now ('color') has lost.
            // The winner is the *other* player ('color.other()').
            PylosPlayerColor winner = color.other();

            // We score this from the perspective of the player who started the search (this.PLAYER_COLOR)
            if (winner == this.PLAYER_COLOR) {
                return Float.POSITIVE_INFINITY - (SEARCH_DEPTH - depth); // Win fast
            } else {
                return Float.NEGATIVE_INFINITY + (SEARCH_DEPTH - depth); // Lose slow
            }
        }

        // Base Case 2: Leaf Node (Search depth limit reached)
        // ... (rest of the function is correct) ...
        if (depth == 0) {
            // evalBoard is always called from the perspective of the root player
            return evalBoard(board, color);
        }

        // --- Recursive Step ---
        List<Action> actionList = new ArrayList<>();
        List<Action> actions = generateActions(board, color, state, actionList);
        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);

        // If no actions are possible (e.g., in a completed state that somehow got here)
        if (actions.isEmpty()) {
            // Re-evaluate the board. This can happen if the terminal state
            // was not caught by the depth == 0 or state == COMPLETED checks
            // (e.g., a board with no moves, but not technically 'COMPLETED' yet).
            return evalBoard(board, color);
        }

        float bestScore = Float.NEGATIVE_INFINITY;

        for (Action action : actions) {
            action.simulate(simulator);

            // Recursive call for the *next* player
            // The score returned is from *their* perspective, so we negate it
            float score = -negamax(board, simulator.getColor(), simulator.getState(), depth - 1, -beta, -alpha);

            action.reverseSimulate(simulator);
            bestScore = Math.max(bestScore, score);
            alpha = Math.max(alpha, score);
            if (alpha >= beta) {
                break; // Beta cutoff
            }
        }

        transpositionTable.put(boardKey, bestScore);
        return bestScore;
    }


    /**
     * UNCHANGED: This is now the leaf node evaluation function for the search.
     * It is always called from the perspective of the player who started the search.
     */
    private float evalBoard(PylosBoard board, PylosPlayerColor color) {
        long boardAsLong = board.toLong();

        //convert board to array of bits
        float[] boardAsArray = new float[60];
        for (int i = 0; i < 60; i++) {
            int leftShifts = 59 - i;
            boolean light = (boardAsLong & (1L << leftShifts)) == 0;
            boardAsArray[i] = light ? 0 : 1;
        }

        float output = Float.NaN;
        try(Tensor inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{boardAsArray}))) {
            try(TFloat32 outputTensor = (TFloat32) model.session().runner()
                    .feed("serving_default_keras_tensor:0", inputTensor)
                    .fetch("StatefulPartitionedCall_1:0")
                    .run().get(0)){
                output = outputTensor.getFloat();
            }
        }

        assert !Float.isNaN(output) : "output is NaN";

        // This switch is crucial: it returns the score from the perspective
        // of the 'color' parameter, which will be this.PLAYER_COLOR.
        return switch (color) {
            case LIGHT -> output; // If I am LIGHT, I want a high output
            case DARK -> -output;  // If I am DARK, I want a low output (so -output is high)
        };
    }


    /**
     * UNCHANGED: Helper method to generate all possible actions
     */
    private static List<Action> generateActions(PylosBoard board, PylosPlayerColor color, PylosGameState state, List<Action> actionList) {
        actionList.clear();
        PylosSphere[] spheres = board.getSpheres(color);

        switch (state) {
            case MOVE -> {
                PylosLocation[] locations = board.getLocations();
                PylosSquare[] squares = board.getAllSquares();
                List<PylosLocation> availableFullSquaresTopLocations = new ArrayList<>();

                // Add actions for moving a sphere to a higher location
                for (PylosSquare square : squares)
                    if (square.getTopLocation().isUsable())
                        availableFullSquaresTopLocations.add(square.getTopLocation());

                for (PylosSphere sphere : spheres)
                    if (!sphere.isReserve())
                        for (PylosLocation location : availableFullSquaresTopLocations)
                            if (sphere.canMoveTo(location) && sphere.getLocation() != location)
                                actionList.add(new Action(ActionType.MOVE, sphere, location));

                // Add actions for moving a reserve sphere to a free location
                for (PylosLocation location : locations)
                    if (location.isUsable())
                        actionList.add(new Action(ActionType.ADD, board.getReserve(color), location));
            }
            case REMOVE_FIRST -> {
                for (PylosSphere sphere : spheres)
                    if (sphere.canRemove())
                        actionList.add(new Action(ActionType.REMOVE_FIRST, sphere, null));
            }
            case REMOVE_SECOND -> {
                actionList.add(new Action(ActionType.PASS, null, null));
                for (PylosSphere sphere : spheres)
                    if (sphere.canRemove())
                        actionList.add(new Action(ActionType.REMOVE_SECOND, sphere, null));
            }
        }

        return actionList;
    }


    /**
     * UNCHANGED: ActionType enum
     */
    enum ActionType {
        ADD,
        MOVE,
        REMOVE_FIRST,
        REMOVE_SECOND,
        PASS
    }

    /**
     * UNCHANGED: Action helper class
     */
    private static class Action {
        private final ActionType type;
        private final PylosSphere sphere;
        private final PylosLocation location;

        private PylosLocation prevLocation;
        private PylosGameState prevState;
        private PylosPlayerColor prevColor;

        public Action(ActionType type, PylosSphere sphere, PylosLocation location) {
            this.type = type;
            this.sphere = sphere;
            this.location = location;
        }

        public void execute(PylosGameIF game) {
            switch (type) {
                case ADD, MOVE ->
                        game.moveSphere(sphere, location);
                case REMOVE_FIRST, REMOVE_SECOND ->
                        game.removeSphere(sphere);
                case PASS ->
                        game.pass();
                default ->
                        throw new IllegalStateException("type not found in switch");
            }
        }

        public void simulate(PylosGameSimulator simulator) {
            prevState = simulator.getState();
            prevColor = simulator.getColor();

            if (type == ActionType.MOVE || type == ActionType.REMOVE_FIRST || type == ActionType.REMOVE_SECOND) {
                // Save the previous location of the sphere
                prevLocation = sphere.getLocation();
                assert prevLocation != null : "prevLocation is null";
            }
            switch (type) {
                case ADD, MOVE ->
                        simulator.moveSphere(sphere, location);
                case REMOVE_FIRST, REMOVE_SECOND ->
                        simulator.removeSphere(sphere);
                case PASS ->
                        simulator.pass();
                default ->
                        throw new IllegalStateException("type not found in switch");
            }
        }

        public void reverseSimulate(PylosGameSimulator simulator) {
            switch (type) {
                case ADD ->
                        simulator.undoAddSphere(sphere, prevState, prevColor);
                case MOVE ->
                        simulator.undoMoveSphere(sphere, prevLocation, prevState, prevColor);
                case REMOVE_FIRST ->
                        simulator.undoRemoveFirstSphere(sphere, prevLocation, prevState, prevColor);
                case REMOVE_SECOND ->
                        simulator.undoRemoveSecondSphere(sphere, prevLocation, prevState, prevColor);
                case PASS ->
                        simulator.undoPass(prevState, prevColor);
                default ->
                        throw new IllegalStateException("type not found in switch");
            }
        }
    }
}

