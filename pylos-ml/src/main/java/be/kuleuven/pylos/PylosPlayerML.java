package be.kuleuven.pylos;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.List;


public class PylosPlayerML extends PylosPlayer {

    private final SavedModelBundle model;
    private static final int SEARCH_DEPTH = 2;

    public PylosPlayerML(SavedModelBundle model) {
        this.model = model;
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
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

    private Action findBestAction(PylosBoard board, PylosPlayerColor color, PylosGameState state) {
        List<Action> actions = new ArrayList<>();
        generateActions(board, color, state, actions);

        float bestScore = Float.NEGATIVE_INFINITY;
        Action bestAction = null;

        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);

        for (Action action : actions) {
            action.simulate(simulator);
            float score = -negamax(board, simulator.getColor(), simulator.getState(), SEARCH_DEPTH - 1, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
            action.reverseSimulate(simulator);

            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
        }

        if (bestAction == null && !actions.isEmpty()) return actions.get(0);
        return bestAction;
    }

    private float negamax(PylosBoard board, PylosPlayerColor color, PylosGameState state, int depth, float alpha, float beta) {
        if (state == PylosGameState.COMPLETED) {
            return Float.NEGATIVE_INFINITY;
        }

        if (depth == 0) {
            return evalBoard(board, color);
        }

        List<Action> actions = new ArrayList<>();
        generateActions(board, color, state, actions);

        if (actions.isEmpty()) return evalBoard(board, color);

        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);
        float bestScore = Float.NEGATIVE_INFINITY;

        for (Action action : actions) {
            action.simulate(simulator);
            float score = -negamax(board, simulator.getColor(), simulator.getState(), depth - 1, -beta, -alpha);
            action.reverseSimulate(simulator);

            bestScore = Math.max(bestScore, score);
            alpha = Math.max(alpha, score);
            if (alpha >= beta) break;
        }
        return bestScore;
    }

    /**
     * Evaluates the board using the TensorFlow model.
     * NEW INPUT SIZE: 37 features
     */
    private float evalBoard(PylosBoard board, PylosPlayerColor color) {
        long boardAsLong = board.toLong();

        // Input Size: 37
        // 0-29: Board Locations
        // 30: My Reserve (normalized)
        // 31: Enemy Reserve (normalized)
        // 32: Reserve Difference
        // 33-36: Layer Scores (Z0, Z1, Z2, Z3)
        float[] inputs = new float[37];

        int lightCount = 0;
        int darkCount = 0;

        // --- 1. PARSE BOARD (Indices 0-29) ---
        for (int loc = 0; loc < 30; loc++) {
            int shift = loc * 2;
            long val = (boardAsLong >> shift) & 3;

            float inputVal = 0.0f;

            if (val == 1) {
                lightCount++;
                inputVal = (color == PylosPlayerColor.LIGHT) ? 1.0f : -1.0f;
            }
            else if (val == 2) {
                darkCount++;
                inputVal = (color == PylosPlayerColor.DARK) ? 1.0f : -1.0f;
            }

            inputs[loc] = inputVal;
        }

        // --- 2. CALCULATE RESERVES (Indices 30-33) ---
        int lightReserves = 15 - lightCount;
        int darkReserves = 15 - darkCount;

        float myReserves, enemyReserves, reserveDiff, materialDiff;

        if (color == PylosPlayerColor.LIGHT) {
            myReserves = lightReserves / 15.0f;
            enemyReserves = darkReserves / 15.0f;
            reserveDiff = (lightReserves - darkReserves) / 15.0f;
        } else {
            myReserves = darkReserves / 15.0f;
            enemyReserves = lightReserves / 15.0f;
            reserveDiff = (darkReserves - lightReserves) / 15.0f;
        }

        inputs[30] = myReserves;
        inputs[31] = enemyReserves;
        inputs[32] = reserveDiff;

        // --- 3. CALCULATE LAYER SCORES (Indices 34-37) ---
        float sumZ0 = 0;
        for(int i=0; i<=15; i++) sumZ0 += inputs[i];
        inputs[33] = sumZ0 / 16.0f;

        float sumZ1 = 0;
        for(int i=16; i<=24; i++) sumZ1 += inputs[i];
        inputs[34] = sumZ1 / 9.0f;

        float sumZ2 = 0;
        for(int i=25; i<=28; i++) sumZ2 += inputs[i];
        inputs[35] = sumZ2 / 4.0f;

        inputs[36] = inputs[29];

        // --- 4. RUN INFERENCE ---
        float output = Float.NaN;
        try (Tensor inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{inputs}))) {
            try (TFloat32 outputTensor = (TFloat32) model.session().runner()
                    .feed("serving_default_keras_tensor:0", inputTensor)
                    .fetch("StatefulPartitionedCall_1:0")
                    .run().get(0)) {
                output = outputTensor.getFloat();
            }
        }
        return output;
    }

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