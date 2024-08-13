package Enemies;


import javafx.scene.shape.Box;

import java.util.*;

import AnimatedObjects.Updatable;
import Game.*;
/**
 * Class which describes the basic requirements for properly working enemy object.
 */
public abstract class Enemy implements Updatable
{
    /**
     * Direction of enemy movement in relation to map.
     */
    public enum Direction {NORTH, SOUTH, EAST, WEST, IDLE}

    /**
     * Method defines which tiles on map an enemy can visit.
     * @return list of characters representing tiles in map.
     */
    abstract List<Character> getAllowedSpaces();

    /**
     * @return direction of moving enemy.
     */
    abstract public Direction getDirection();

    /**
     * @return x coordinate of enemy in scene.
     */
    abstract public double getX();

    /**
     * @return y coordinate of enemy in scene.
     */
    abstract public double getY();

    /**
     * @return z coordinate of enemy in scene.
     */
    abstract public double getZ();

    /**
     * @return width of image representing enemy.
     */
    abstract public double getWidth();

    /**
     * @return height of image representing enemy.
     */
    abstract public double getHeight();

    /**
     * @return the max level of breadth first search when searching for player on map.
     */
    abstract public int getMaxDistance();

    /**
     * @return the movement speed of enemy
     */
    abstract public double getSpeed();

    /**
     * @return box representing space occupied by enemy in scene.
     */
    abstract public Box getHIT_BOX();

    /**
     * Method deals damage caused by enemy to player.
     * @param damage amount of damage dealt by enemy.
     */
    abstract public void dealDamage(double damage);

    /**
     * Method updates enemy position in scene and plays appropriate animation.
     */
    abstract public void update();

    /**
     * Checks if row and column are proper indexes relative to size of map and it's rows.
     * @param row row of map.
     * @param column column of map.
     * @return Whether specified tile on map is an tile which enemy can move on to.
     */
    boolean freeSpace(int row, int column)
    {
        char[][] map = Game.getMap();
        if (map == null)
            return false;
        if (row < 0 || row >= map.length)
            return false;
        if (map[row] == null)
            return false;
        if (column < 0 || column >= map[row].length)
            return false;

        return getAllowedSpaces().contains(map[row][column]);
    }

    /**
     * @param direction direction
     * @return displacement on X axis in relation to Scene.
     */
    public double dx(Direction direction)
    {
        return switch (direction)
                {
                    case EAST -> getSpeed();
                    case WEST -> -getSpeed();
                    default -> 0;
                };
    }

    /**
     * @param direction direction
     * @return displacement on z axis in relation to Scene.
     */
    public double dz(Direction direction)
    {
        return switch (direction)
                {
                    case NORTH -> getSpeed();
                    case SOUTH -> -getSpeed();
                    default -> 0;
                };
    }

    /**
     * Finds the row and column of upper left corner of enemy and bottom right corner.
     * If the rows and columns are different then enemy is not centered on tile.
     * @return Whether enemy is located in center of tile.
     */
    public boolean adjusted()
    {
        int columnA = (int)Math.floor((getX() - (getWidth() / 2))/ Game.UNIT);
        int rowA = (int)Math.floor((getZ() - (getWidth() / 2)) / Game.UNIT);
        int columnB = (int)Math.floor((getX() + (getWidth() / 2)) / Game.UNIT);
        int rowB = (int)Math.floor((getZ() + (getWidth() / 2)) / Game.UNIT);

        return columnA == columnB && rowA == rowB;
    }

    /**
     * Finds if tow parallel lines with lineOfSight, both being half of enemy width distant from line of sight,
     * do not pass trough tile which enemy can not move onto.
     * Line of sight is line between enemy and player coordinates.
     * @return whether an enemy can see a player.
     */
    public boolean lineOfSight()
    {
        double xs = getX(), xp = Player.getX();
        double zs = getZ(), zp = Player.getZ();
        double k = (zp - zs) / (xp - xs);
        double alpha = Math.atan(k);
        double dz = (getWidth() / 2) / Math.cos(alpha);

        double x = xs;
        while (true)
        {
            if (xs > xp && x < xp)
                break;
            if (xs <= xp && x > xp)
                break;

            int column = (int)Math.floor(x / Game.UNIT);

            double z = k * (x - xs) + zs + dz;
            int row = (int)Math.floor(z / Game.UNIT);

            if ( ! freeSpace(row, column))
                return false;

            z = k * (x - xs) + zs - dz;
            row = (int)Math.floor(z / Game.UNIT);

            if ( ! freeSpace(row, column))
                return false;

            if (xs > xp)
                x -= (Game.UNIT / 4d);
            else
                x += (Game.UNIT / 4d);
        }
        return true;
    }

    /**
     * Searches for player row and column on map using breadth first algorithm, tracing trough tiles an enemy can visit.
     * @param row starting row on map.
     * @param column starting column on map.
     * @param level starting level of breadth search.
     * @return level of the first map tile satisfying search condition or max level if tile wan not found.
     */
    public int findPlayer(int row, int column, int level)
    {
        if (! freeSpace(row, column))
            return getMaxDistance();

        Queue<List<Integer>> queue = new ArrayDeque<>();
        Map<Integer, Stack<Integer>> visited = new HashMap<>();
        queue.add(List.of(row, column, level));

        int playerColumn = (int)Math.floor(Player.getX() / Game.UNIT);
        int playerRow = (int)Math.floor(Player.getZ() / Game.UNIT);

        while ( ! queue.isEmpty())
        {
            List<Integer> coords = queue.remove();
            row = coords.get(0);
            column = coords.get(1);
            level = coords.get(2);

            if (level >= getMaxDistance())
                break;

            if (row == playerRow && column == playerColumn)
                return level;

            if ( ! visited.containsKey(row))
                visited.put(row, new Stack<>());

            if ( ! visited.get(row).contains(column))
                visited.get(row).add(column);

            for (int newRow : new int[] { row - 1, row + 1 })
            {
                if (! freeSpace(newRow, column))
                    continue;

                if ( ! visited.containsKey(newRow))
                    visited.put(newRow, new Stack<>());

                if (visited.get(newRow).contains(column))
                    continue;

                visited.get(newRow).add(column);
                queue.add(List.of(newRow, column, level + 1));
            }

            for (int newColumn : new int[] { column - 1, column + 1 })
            {
                if (! freeSpace(row, newColumn))
                    continue;

                if ( ! visited.containsKey(row))
                    visited.put(row, new Stack<>());

                if (visited.get(row).contains(newColumn))
                    continue;

                visited.get(row).add(newColumn);
                queue.add(List.of(row, newColumn, level + 1));
            }

        }
        return getMaxDistance();
    }

    /**
     * Tests all direction and returns the direction which would move enemy closer to player.
     * @return direction of movement
     */
    public Direction findDirection()
    {
        int column = (int)Math.floor((getX())/ Game.UNIT);
        int row = (int)Math.floor((getZ()) / Game.UNIT);

        if ( ! adjusted())
            return getDirection();

        int distance = getMaxDistance();
        Direction direction = Direction.IDLE;

        int result = findPlayer(row - 1, column, 1);
        if (result < distance)
        {
            direction = Direction.SOUTH;
            distance = result;
        }
        result = findPlayer(row, column - 1, 1);
        if (result < distance)
        {
            direction = Direction.WEST;
            distance = result;
        }
        result = findPlayer(row, column + 1, 1);
        if (result < distance)
        {
            direction = Direction.EAST;
            distance = result;
        }
        result = findPlayer(row + 1, column, 1);
        if (result < distance)
        {
            direction = Direction.NORTH;
            distance = result;
        }
        result = findPlayer(row, column, 0);
        if (result < distance)
        {
            direction = Direction.IDLE;
        }
        return direction;
    }
}
