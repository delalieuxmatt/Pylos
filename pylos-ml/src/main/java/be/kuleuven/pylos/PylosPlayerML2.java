package be.kuleuven.pylos;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.List;

public class PylosPlayerML2 extends PylosPlayer {

    private final SavedModelBundle model;

    public PylosPlayerML2(SavedModelBundle model) {
        this.model = model;
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        // Just find the best immediate move according to the Neural Net
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.MOVE);
        bestAction.execute(game);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_FIRST);
        bestAction.execute(game);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        Action bestAction = findBestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_SECOND);
        bestAction.execute(game);
    }

    /**
     * PURE ML SEARCH STRATEGY:
     * 1. Generate all legal moves.
     * 2. Simulate each move to see the resulting board.
     * 3. Ask the Neural Network: "How good is this board for ME?"
     * 4. Pick the move with the highest score.
     */
    private Action findBestAction(PylosBoard board, PylosPlayerColor color, PylosGameState state) {
        List<Action> actionList = new ArrayList<>();
        List<Action> actions = generateActions(board, color, state, actionList);
        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);

        Action bestAction = null;
        float bestEval = Float.NEGATIVE_INFINITY;

        for (Action action : actions) {
            // 1. Simulate the move
            action.simulate(simulator);

            // 2. Evaluate the board AFTER the move.
            // We ask: "What is the value of this board for ME (this.PLAYER_COLOR)?"
            float eval = evalBoard(board, this.PLAYER_COLOR);

            // 3. Undo the simulation
            action.reverseSimulate(simulator);

            // 4. Maximize the score
            if (eval > bestEval) {
                bestEval = eval;
                bestAction = action;
            }
        }

        // Fallback if list is empty or errors occur
        if (bestAction == null && !actions.isEmpty()) {
            return actions.get(0);
        }

        return bestAction;
    }

    /**
     * Evaluates the board using the TensorFlow model.
     * Contains the logic to flip the input perspective if the player is DARK.
     */
    private float evalBoard(PylosBoard board, PylosPlayerColor color) {
        long boardAsLong = board.toLong();

        // 1. Prepare the input array (60 inputs)
        float[] boardAsArray = new float[60];
        for (int i = 0; i < 60; i++) {
            int leftShifts = 59 - i;
            // In standard Pylos logic: 0 = Light, 1 = Dark (in the bitboard bits)
            boolean isLightSphere = (boardAsLong & (1L << leftShifts)) == 0;

            float val;
            // --- PERSPECTIVE LOGIC ---
            if (color == PylosPlayerColor.LIGHT) {
                // If I am LIGHT, the model expects My Pieces = 0, Enemy = 1
                val = isLightSphere ? 0.0f : 1.0f;
            } else {
                // If I am DARK, the model expects My Pieces = 0, Enemy = 1
                // So we flip the inputs: Light spheres become "Enemy" (1.0)
                val = isLightSphere ? 1.0f : 0.0f;
            }
            boardAsArray[i] = val;
        }

        float output = Float.NaN;
        try (Tensor inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{boardAsArray}))) {
            try (TFloat32 outputTensor = (TFloat32) model.session().runner()
                    .feed("serving_default_keras_tensor:0", inputTensor)
                    .fetch("StatefulPartitionedCall_1:0")
                    .run().get(0)) {
                output = outputTensor.getFloat();
            }
        }

        // 2. Return output directly.
        // The model is trained to return Higher Scores = Better for the "Input Player".
        return output;
    }

    /**
     * Helper: Generate all possible actions
     */
    private static List<Action> generateActions(PylosBoard board, PylosPlayerColor color, PylosGameState state, List<Action> actionList) {
        actionList.clear();
        PylosSphere[] spheres = board.getSpheres(color);

        switch (state) {
            case MOVE -> {
                PylosLocation[] locations = board.getLocations();
                PylosSquare[] squares = board.getAllSquares();
                List<PylosLocation> availableFullSquaresTopLocations = new ArrayList<>();

                for (PylosSquare square : squares)
                    if (square.getTopLocation().isUsable())
                        availableFullSquaresTopLocations.add(square.getTopLocation());

                for (PylosSphere sphere : spheres)
                    if (!sphere.isReserve())
                        for (PylosLocation location : availableFullSquaresTopLocations)
                            if (sphere.canMoveTo(location) && sphere.getLocation() != location)
                                actionList.add(new Action(ActionType.MOVE, sphere, location));

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

    enum ActionType { ADD, MOVE, REMOVE_FIRST, REMOVE_SECOND, PASS }

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
                case ADD, MOVE -> game.moveSphere(sphere, location);
                case REMOVE_FIRST, REMOVE_SECOND -> game.removeSphere(sphere);
                case PASS -> game.pass();
            }
        }

        public void simulate(PylosGameSimulator simulator) {
            prevState = simulator.getState();
            prevColor = simulator.getColor();
            if (type == ActionType.MOVE || type == ActionType.REMOVE_FIRST || type == ActionType.REMOVE_SECOND) {
                prevLocation = sphere.getLocation();
            }
            switch (type) {
                case ADD, MOVE -> simulator.moveSphere(sphere, location);
                case REMOVE_FIRST, REMOVE_SECOND -> simulator.removeSphere(sphere);
                case PASS -> simulator.pass();
            }
        }

        public void reverseSimulate(PylosGameSimulator simulator) {
            switch (type) {
                case ADD -> simulator.undoAddSphere(sphere, prevState, prevColor);
                case MOVE -> simulator.undoMoveSphere(sphere, prevLocation, prevState, prevColor);
                case REMOVE_FIRST -> simulator.undoRemoveFirstSphere(sphere, prevLocation, prevState, prevColor);
                case REMOVE_SECOND -> simulator.undoRemoveSecondSphere(sphere, prevLocation, prevState, prevColor);
                case PASS -> simulator.undoPass(prevState, prevColor);
            }
        }
    }
}