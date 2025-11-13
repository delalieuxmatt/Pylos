package be.kuleuven.pylos.player.student;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosGameSimulator;
import be.kuleuven.pylos.game.PylosGameState;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.game.PylosSquare;
import be.kuleuven.pylos.player.PylosPlayer;


public class StudentPlayerTano extends PylosPlayer {


    private static final double ROOT_TIE_DELTA = 0.0; //bepaald hoe groot het genomen interval van beste scores is voor random te kiezen

    private static final double WIN_SCORE = 1_000_000.0;
    private static final double LOSS_SCORE = -WIN_SCORE;

    /* high bits to distinguish state+color in hash */
    private static final long COLOR_FLAG = 1L << 60;
    private static final long MOVE_FLAG = 1L << 61;
    private static final long REMOVE_FIRST_FLAG = 1L << 62;
    private static final long REMOVE_SECOND_FLAG = 1L << 63;

    private static final int RESERVE_WEIGHT = 600;
    private static final int THREAT_WEIGHT = 180;
    private static final int STRUCTURE_WEIGHT = 35;
    private static final int CENTRAL_WEIGHT = 4;
    private static final int HEIGHT_WEIGHT = 30;
    private static final int MOBILITY_WEIGHT = 25;
    private static final int REMOVABLE_WEIGHT = 20;

    private PylosGameSimulator simulator;
    private PylosBoard board;
    private PylosLocation[] allLocations;
    private PylosSquare[] allSquares;
    private Map<Long, TranspositionEntry> transpositionTable;
    private int currentDepthLimit;

    private Action bestAction;

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        prepareSearch(game.getState(), board);
        Action action = decideBestAction();
        executeAction(game, action);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        prepareSearch(game.getState(), board);
        Action action = decideBestAction();
        executeAction(game, action);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        prepareSearch(game.getState(), board);
        Action action = decideBestAction();
        executeAction(game, action);
    }

    private void executeAction(PylosGameIF game, Action action) {
        if (action == null) {
            throw new IllegalStateException("Geen actie gevonden voor state " + simulator.getState());
        }
        switch (action.type) {
            case MOVE:
                game.moveSphere(action.sphere, action.to);
                break;
            case REMOVE:
                game.removeSphere(action.sphere);
                break;
            case PASS:
                game.pass();
                break;
            default:
                throw new IllegalStateException("Onbekend actietype " + action.type);
        }
    }

    private void prepareSearch(PylosGameState state, PylosBoard board) {
        this.board = board;
        this.simulator = new PylosGameSimulator(state, PLAYER_COLOR, board);
        this.allLocations = board.getLocations();
        this.allSquares = board.getAllSquares();
        this.transpositionTable = new HashMap<>(1 << 16);
        this.bestAction = null;
    }

    private Action decideBestAction() {
        int depthMax = computeDepthLimit(); //kijkt hoeveel bollen op het bord -> zo diepte
        Action fallback = null;

        for (int depth = 1; depth <= depthMax; depth++) {
            currentDepthLimit = depth;
            bestAction = null;
            double score = searchRoot(depth);
            if (bestAction != null) {
                fallback = bestAction;
            }
            if (Math.abs(score) >= WIN_SCORE - 1000) { //als uitkomst al vast ligt vroegtijdig stoppen
                break; // forceert gevonden resultaat
            }
        }

        if (fallback == null) {
            List<Action> candidates = generateActions(simulator.getState(), simulator.getColor());
            if (!candidates.isEmpty()) {
                fallback = candidates.get(0);
            }
        }

        if (fallback == null) {
            throw new IllegalStateException("Geen legale acties beschikbaar");
        }
        return fallback;
    }

    private int computeDepthLimit() {
        int onBoard = board.getNumberOfSpheresOnBoard();
        if (onBoard < 10) {
            return 7;
        } else if (onBoard < 16) {
            return 8;
        } else if (onBoard < 22) {
            return 9;
        } else if (onBoard < 26) {
            return 10;
        }
        return 11;
    }

    private double searchRoot(int depthLimit) { //minimax: neemt lijst van acties van mijn kleur en kijkt welke beste
        if (simulator.getColor() != PLAYER_COLOR) {
            throw new IllegalStateException("Root search verwacht eigen kleur aan zet");
        } //maakt lijst van mogelijke acties uit de situatie
        List<Action> actions = generateActions(simulator.getState(), simulator.getColor());
        if (actions.isEmpty()) {
            return evaluate();
        }

        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;
        double bestValue = Double.NEGATIVE_INFINITY; //beste zo klein mogelijk zo is altijd eerste een zet beter

        class AV {
            final Action a;
            final double v;
            AV(Action a,double v){
                this.a=a; this.v=v;
            }
        }
        List<AV> scored = new ArrayList<>(actions.size());

        //doorloopt lijst met acties en kijkt dan volgens de evalutie welke score de zet krijgt
        for (Action action : actions) {
            double v = applyActionAndSearch(action, depthLimit, alpha, beta);
            scored.add(new AV(action, v));
            if (v > bestValue ) {
                bestValue = v;

            }

            alpha = Math.max(alpha, bestValue);
            if (alpha >= beta) {
                break; //pruning
            }

        }

        List<Action> pool = new ArrayList<>();
        for (AV av : scored) if (av.v >= bestValue - ROOT_TIE_DELTA) pool.add(av.a);

        if (!pool.isEmpty()){
            java.util.Collections.shuffle(pool, getRandom());
            bestAction = pool.get(0);
        }

        return bestValue;
    }

    private double search(int depth, double alpha, double beta) {
        PylosGameState state = simulator.getState();

        if (state == PylosGameState.COMPLETED) {
            return simulator.getWinner() == PLAYER_COLOR
                    ? WIN_SCORE - (currentDepthLimit - depth) //snelwinnen bv win in 3 beter dan win in 5
                    : LOSS_SCORE + (currentDepthLimit - depth); //traagverliezen bv verlies in 5 beter dan verlies in 3
        }
        if (state == PylosGameState.DRAW) {
            return 0;
        }

        if (depth == 0) {
            return evaluate();
        }

        boolean maximizing = simulator.getColor() == PLAYER_COLOR;
        long key = encodeState(state, simulator.getColor());
        TranspositionEntry entry = transpositionTable.get(key);
        //als de move al in de tabel zit en de depth van de entry in de tabel is groter dan die dat we nu op zitten
        if (entry != null && entry.depth >= depth) {
            if (entry.type == EntryType.EXACT) {
                return entry.value; //zelfde geef waarde uit tabel
            } else if (entry.type == EntryType.LOWER_BOUND) {
                alpha = Math.max(alpha, entry.value); //als
            } else if (entry.type == EntryType.UPPER_BOUND) {
                beta = Math.min(beta, entry.value);
            }
            if (alpha >= beta) {
                return entry.value;
            }
        }

        List<Action> actions = generateActions(state, simulator.getColor());
        if (actions.isEmpty()) {
            return evaluate();
        }

        double originalAlpha = alpha;
        double originalBeta = beta;
        double best = maximizing ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        for (Action action : actions) {
            double value = applyActionAndSearch(action, depth, alpha, beta);
            if (maximizing) {
                if (value > best) {
                    best = value;
                }
                alpha = Math.max(alpha, best);
            } else {
                if (value < best) {
                    best = value;
                }
                beta = Math.min(beta, best);
            }
            if (alpha >= beta) { //pruning
                break;
            }
        }

        EntryType type;
        if (best <= originalAlpha) { //als beste kleiner dan orginele aplha zeggen we upper_bound
            type = EntryType.UPPER_BOUND;
        } else if (best >= originalBeta) { //als beter dan orginele beta zeggen lower_bound
            type = EntryType.LOWER_BOUND;
        } else {
            type = EntryType.EXACT;
        }
        transpositionTable.put(key, new TranspositionEntry(depth, best, type));
        return best;
    }

    private double applyActionAndSearch(Action action, int depth, double alpha, double beta) {
        PylosGameState prevState = simulator.getState();
        PylosPlayerColor prevColor = simulator.getColor();
        double result;
        switch (action.type) {
            case MOVE: //beweegt de sphere of zet in
                if (action.fromReserve) {
                    simulator.moveSphere(action.sphere, action.to);
                    result = search(depth - 1, alpha, beta);
                    simulator.undoAddSphere(action.sphere, prevState, prevColor);
                } else {
                    simulator.moveSphere(action.sphere, action.to);
                    result = search(depth - 1, alpha, beta);
                    simulator.undoMoveSphere(action.sphere, action.from, prevState, prevColor);
                }
                break;
            case REMOVE:
                simulator.removeSphere(action.sphere);
                result = search(depth - 1, alpha, beta);
                if (prevState == PylosGameState.REMOVE_FIRST) {
                    simulator.undoRemoveFirstSphere(action.sphere, action.from, prevState, prevColor);
                } else {
                    simulator.undoRemoveSecondSphere(action.sphere, action.from, prevState, prevColor);
                }
                break;
            case PASS:
                simulator.pass();
                result = search(depth - 1, alpha, beta);
                simulator.undoPass(prevState, prevColor);
                break;
            default:
                throw new IllegalStateException("Onbekend actietype " + action.type);
        }
        return result;
    }

    private List<Action> generateActions(PylosGameState state, PylosPlayerColor color) {
        List<Action> actions = new ArrayList<>();
        switch (state) {
            case MOVE:
                generateMoveActions(color, actions);
                break;
            case REMOVE_FIRST:
                generateRemoveActions(color, state, actions);
                break;
            case REMOVE_SECOND:
                generateRemoveActions(color, state, actions);
                actions.add(new Action(ActionType.PASS, null, null, null, false, -50));
                break;
            case COMPLETED:
            case DRAW:
            case ABORTED:
                break;
            default:
                throw new IllegalStateException("State niet ondersteund: " + state);
        }
        if (!actions.isEmpty()) {
            actions.sort((a, b) -> Integer.compare(b.orderScore, a.orderScore));
        }
        return actions;
    }

    private void generateMoveActions(PylosPlayerColor color, List<Action> actions) {
        if (board.getReservesSize(color) > 0) {
            PylosSphere reserve = board.getReserve(color);
            for (PylosLocation location : allLocations) {
                if (location.isUsable()) {
                    int score = moveOrderingScore(reserve, null, location, color, true);
                    actions.add(new Action(ActionType.MOVE, reserve, null, location, true, score));
                }
            }
        }

        PylosSphere[] spheres = board.getSpheres(color);
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve() && sphere.canMove()) {
                PylosLocation from = sphere.getLocation();
                for (PylosLocation location : allLocations) {
                    if (sphere.canMoveTo(location)) {
                        int score = moveOrderingScore(sphere, from, location, color, false);
                        actions.add(new Action(ActionType.MOVE, sphere, from, location, false, score));
                    }
                }
            }
        }
    }

    private void generateRemoveActions(PylosPlayerColor color, PylosGameState state, List<Action> actions) {
        PylosSphere[] spheres = board.getSpheres(color);
        for (PylosSphere sphere : spheres) {
            if (sphere.canRemove()) {
                PylosLocation from = sphere.getLocation();
                int score = removalOrderingScore(from, state);
                actions.add(new Action(ActionType.REMOVE, sphere, from, null, false, score));
            }
        }
    }

    private int moveOrderingScore(PylosSphere sphere, PylosLocation from, PylosLocation to, PylosPlayerColor color, boolean fromReserve) {
        int score = 0;
        for (PylosSquare square : to.getSquares()) {
            int own = square.getInSquare(color);
            int opp = square.getInSquare(color.other());
            if (own == 3 && opp == 0) {
                score += 400; // completes square
            } else if (opp == 3 && own == 0) {
                score += 250; // blocks opponent square
            } else if (own == 2 && opp == 0) {
                score += 80;
            }
            if (square.getTopLocation() != null && square.getTopLocation().isUsable() && own == 4) {
                score += 60; // prepares top placement
            }
        }
        if (!fromReserve && from != null) {
            for (PylosSquare square : from.getSquares()) {
                int own = square.getInSquare(color);
                int opp = square.getInSquare(color.other());
                if (own >= 2 && opp == 0) { //tegen verplaatsen van plaats waar mogelijke square
                    score -= own * 35; // je wil geen basis verwijderen
                }
            }
            score += (to.Z - from.Z) * 50;//hoogte is goed
            score += (centrality(to) - centrality(from)) * 2;// hoe meer in het centrum hoe beter
        } else { //als sphere wordt toegevoegd
            score += to.Z * 45;
            score += centrality(to) * 2;
        }
        return score;
    }

    private int removalOrderingScore(PylosLocation from, PylosGameState state) {
        int score = from.Z * 80 - centrality(from) * 3;
        if (state == PylosGameState.REMOVE_FIRST) {
            for (PylosSquare square : from.getSquares()) {
                if (square.getInSquare(PLAYER_COLOR) >= 3) {
                    score -= 100; //beschermt net gebouwde structuren
                }
            }
        }
        return score;
    }

    private double evaluate() {
        PylosPlayerColor opponent = PLAYER_COLOR.other();
        int reserveDiff = board.getReservesSize(PLAYER_COLOR) - board.getReservesSize(opponent); //verschil in resterve spheres

        int myThreats = 0;
        int oppThreats = 0;
        int myStructure = 0;
        int oppStructure = 0;

        for (PylosSquare square : allSquares) {
            int mine = square.getInSquare(PLAYER_COLOR);
            int opp = square.getInSquare(opponent);

            if (mine == 4) {
                myThreats += 2;
            }
            if (opp == 4) {
                oppThreats += 2;
            }
            if (mine == 3 && opp == 0) {
                myThreats++;
            }
            if (opp == 3 && mine == 0) {
                oppThreats++;
            }
            if (mine > 0 && opp == 0) {
                myStructure += mine;
            } else if (opp > 0 && mine == 0) {
                oppStructure += opp;
            }
        }

        int myCentral = 0;
        int oppCentral = 0;
        int myHeight = 0;
        int oppHeight = 0;
        int myMobility = 0;
        int oppMobility = 0;
        int myRemovable = 0;
        int oppRemovable = 0;

        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            PylosLocation location = sphere.getLocation();
            if (location != null) {
                myCentral += centrality(location);
                myHeight += location.Z;
                if (!location.hasAbove()) {
                    myMobility++;
                    if (sphere.canRemove()) {
                        myRemovable++;
                    }
                }
            }
        }
        for (PylosSphere sphere : board.getSpheres(opponent)) {
            PylosLocation location = sphere.getLocation();
            if (location != null) {
                oppCentral += centrality(location);
                oppHeight += location.Z;
                if (!location.hasAbove()) {
                    oppMobility++;
                    if (sphere.canRemove()) {
                        oppRemovable++;
                    }
                }
            }
        }

        double score = reserveDiff * RESERVE_WEIGHT
                + (myThreats - oppThreats) * THREAT_WEIGHT //wilt meer squares dan opp
                + (myStructure - oppStructure) * STRUCTURE_WEIGHT //wilt meer squares dan opp
                + (myCentral - oppCentral) * CENTRAL_WEIGHT //meer centrale spheres dan opp
                + (myHeight - oppHeight) * HEIGHT_WEIGHT //meer hoge spheres
                + (myMobility - oppMobility) * MOBILITY_WEIGHT //meer mogelijke spheres om te verplaatsen
                + (myRemovable - oppRemovable) * REMOVABLE_WEIGHT; //meer mogelijke spheres om te verwijderen

        return score;
    }

    private int centrality(PylosLocation location) {
        int effectiveSize = board.SIZE - location.Z;
        int center = (effectiveSize - 1) / 2;
        int dx = Math.abs(location.X - center);
        int dy = Math.abs(location.Y - center);
        int closeness = effectiveSize - (dx + dy);
        return Math.max(closeness, 0);
    }

    //neemt de state en color van speler en kijkt zo wat de
    private long encodeState(PylosGameState state, PylosPlayerColor color) {
        long key = board.toLong();
        if (color == PylosPlayerColor.DARK) {
            key |= COLOR_FLAG; // key = key of colorflag
        }

        switch (state) {
            case MOVE:
                key |= MOVE_FLAG;
                break;
            case REMOVE_FIRST:
                key |= REMOVE_FIRST_FLAG;
                break;
            case REMOVE_SECOND:
                key |= REMOVE_SECOND_FLAG;
                break;
            case COMPLETED:
                break;
            default:
                break;
        }
        return key;
    }

    private enum ActionType {
        MOVE,
        REMOVE,
        PASS
    }

    private enum EntryType {
        EXACT,
        LOWER_BOUND,
        UPPER_BOUND
    }

    private static final class Action {
        final ActionType type;
        final PylosSphere sphere;
        final PylosLocation from;
        final PylosLocation to;
        final boolean fromReserve;
        final int orderScore;

        Action(ActionType type, PylosSphere sphere, PylosLocation from, PylosLocation to, boolean fromReserve, int orderScore) {
            this.type = type;
            this.sphere = sphere;
            this.from = from;
            this.to = to;
            this.fromReserve = fromReserve;
            this.orderScore = orderScore;
        }
    }

    private static final class TranspositionEntry {
        final int depth;
        final double value;
        final EntryType type;

        TranspositionEntry(int depth, double value, EntryType type) {
            this.depth = depth;
            this.value = value;
            this.type = type;
        }
    }
}
