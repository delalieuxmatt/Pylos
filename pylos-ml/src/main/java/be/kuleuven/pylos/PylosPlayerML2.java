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
        Action bestAction = bestAction(board, this.PLAYER_COLOR, PylosGameState.MOVE);
        bestAction.execute(game);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        Action bestAction = bestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_FIRST);
        bestAction.execute(game);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        Action bestAction = bestAction(board, this.PLAYER_COLOR, PylosGameState.REMOVE_SECOND);
        bestAction.execute(game);
    }

    // First iteration of minimax function which returns an action instead of a score
    private Action bestAction(PylosBoard board, PylosPlayerColor color, PylosGameState state) {
        List<Action> actions = generateActions(board, color, state);
        PylosGameSimulator simulator = new PylosGameSimulator(state, color, board);

        Action bestAction = null;
        float bestEval = Float.NEGATIVE_INFINITY;
        for (Action action : actions) {
            action.simulate(simulator);
            float eval = evalBoard(board, color);
            action.reverseSimulate(simulator);
            if (eval > bestEval) {
                bestEval = eval;
                bestAction = action;
            }
        }

        return bestAction;
    }

    // Returns a value which we try to maximise and our opponent tries to minimize.
    private float evalBoard(PylosBoard board, PylosPlayerColor color) {
        long boardAsLong = board.toLong();

        // Input Size: 37
        // 0-29: Board Locations
        // 30: My Reserve (normalized)
        // 31: Enemy Reserve (normalized)
        // 32: Reserve Difference
        // 33-36: Layer Scores (Z0, Z1, Z2, Z3)
        // 37: Square potential
        float[] inputs = new float[38];

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

        float myReserves, enemyReserves, reserveDiff;

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

        inputs[37] = evaluateSquarePotential(board);


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

    private float evaluateSquarePotential(PylosBoard board) {
        float score = 0;
        for (int level = 0; level < 3; level++) {
            int size = 4 - level;
            for (int x = 0; x < size - 1; x++) {
                for (int y = 0; y < size - 1; y++) {
                    score += (float) evaluateSingleSquare(x, y, level, board);
                }
            }
        }
        return score;
    }

    private double evaluateSingleSquare(int x, int y, int level, PylosBoard board) {
        int myCount = 0, oppCount = 0;
        int[][] offsets = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
        for (int[] offset : offsets) {
            PylosLocation loc = board.getBoardLocation(x + offset[0], y + offset[1], level);
            if (loc != null && loc.getSphere() != null) {
                if (loc.getSphere().PLAYER_COLOR == PLAYER_COLOR) myCount++; else oppCount++;
            }
        }
        if (oppCount == 0) {
            if (myCount == 3) return 5.0;
            if (myCount == 2) return 2.0;
        }
        if (myCount == 0) {
            if (oppCount == 3) return -5.0;
            if (oppCount == 2) return -2.0;
        }
        return 0.0;
    }

    private static List<Action> generateActions(PylosBoard board, PylosPlayerColor color, PylosGameState state) {
        List<Action> actions = new ArrayList<>();
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
                                actions.add(new Action(ActionType.MOVE, sphere, location));

                // Add actions for moving a reserve sphere to a free location
                for (PylosLocation location : locations)
                    if (location.isUsable())
                        actions.add(new Action(ActionType.ADD, board.getReserve(color), location));
            }
            case REMOVE_FIRST -> {
                for (PylosSphere sphere : spheres)
                    if (sphere.canRemove())
                        actions.add(new Action(ActionType.REMOVE_FIRST, sphere, null));
            }
            case REMOVE_SECOND -> {
                actions.add(new Action(ActionType.PASS, null, null));
                for (PylosSphere sphere : spheres)
                    if (sphere.canRemove())
                        actions.add(new Action(ActionType.REMOVE_SECOND, sphere, null));
            }
        }

        return actions;
    }


    enum ActionType {
        ADD,
        MOVE,
        REMOVE_FIRST,
        REMOVE_SECOND,
        PASS
    }

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

