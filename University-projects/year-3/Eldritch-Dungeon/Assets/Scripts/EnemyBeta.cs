using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Direction
{
    north,
    northEast,
    east,
    southEast,
    south,
    southWest,
    west,
    nortWest,
    player,
    none
}

public class EnemyBeta : MonoBehaviour
{
    public float health = 100f;
    public float damage = 10f;
    public float horror = 5f;
    public float speed = 0.2f;

    public int perception = 12;

    private List<Direction> availableDirections = new List<Direction>()
    {
        Direction.north,
        Direction.east,
        Direction.south,
        Direction.west,
    };
    private int directionIndex = 0;
    private double time = 0d;
    private (int row, int column) playerPos;
    public SpiderData data;

    private Queue<((int row, int column) goal, (int row, int column) lastPosition, int level)> pathQueue = new Queue<((int, int), (int, int), int)>();
    private List<(int row, int column)> visited = new List<(int, int)>();
    private (int row, int column) goal = (-1, -1);
    private (int row, int column) start = (-1, -1);


    // Start is called before the first frame update
    void Start()
    {
        SetValues();
        time = Random.Range(0, 3);
        int playerCol = Mathf.RoundToInt(Player.x / Map.scale);
        int playerRow = Mathf.RoundToInt(Player.z / Map.scale);
        int col = Mathf.RoundToInt(transform.position.x / Map.scale);
        int row = Mathf.RoundToInt(transform.position.z / Map.scale);
        playerPos = (playerRow, playerCol);
        start = (row, col);
    }

    // Update is called once per frame
    void Update()
    {
        Direction direction = GetDirectionFromPath();
        UpdatePath();
        Move(direction);
    }

    public bool PlayerPositionChanged()
    {
        int playerCol = Mathf.RoundToInt(Player.x / Map.scale);
        int playerRow = Mathf.RoundToInt(Player.z / Map.scale);

        int playerColPrev = playerPos.column;
        int playerRowPrev = playerPos.row;

        if (playerCol != playerColPrev || playerRow != playerRowPrev)
        {
            playerPos = (playerRow, playerCol);
            return true;
        }    
        return false;
    }

    public Direction GetDirectionFromPath()
    {
        if (goal == (-1, -1))
            return Direction.none;

        return GetDirection(start.row, start.column, goal.row, goal.column);
    }

    public void UpdatePath()
    {
        int col = Mathf.RoundToInt(transform.position.x / Map.scale);
        int row = Mathf.RoundToInt(transform.position.z / Map.scale);

        if (goal == (-1, -1))
        {
            goal = FindGoal();
        }

        if (row == goal.row && col == goal.column && IsCentered())
        {
            start = goal;
            goal = FindGoal();
        }
    }

    public (int, int) FindGoal()
    {
        (int, int) square = (-1, -1);

        square = PathToPlayer();
        if (square == (-1, -1))
            square = Patrol();

        return square;
    }

    public (int, int) Patrol()
    {
        Direction direction = availableDirections[directionIndex];
        
        (int row, int col) nextRowCol = GetRowAndColumn(start.row, start.column, direction);
        string cell = Map.GetCell(nextRowCol.row, nextRowCol.col);
        if (CanVisit(cell) && !WallCollision(direction))
            return nextRowCol;    
        else
            directionIndex = (directionIndex + 1) % availableDirections.Count;

        return (-1, -1);
    }

    public (int, int) PathToPlayer()
    {
        int col = Mathf.RoundToInt(transform.position.x / Map.scale);
        int row = Mathf.RoundToInt(transform.position.z / Map.scale);
        int goal_col = Mathf.RoundToInt(Player.x / Map.scale);
        int goal_row = Mathf.RoundToInt(Player.z / Map.scale);

        if (row == goal_row && col == goal_col)
            return (goal_row, goal_col);

        pathQueue.Clear();
        visited.Clear();

        foreach (Direction direction in availableDirections)
        {
            (int row, int col) nextRowCol = GetRowAndColumn(row, col, direction);
            string cell = Map.GetCell(nextRowCol.row, nextRowCol.col);
            if (CanVisit(cell) && !WallCollision(direction))
            {
                pathQueue.Enqueue((nextRowCol, nextRowCol, 1));
            }    
        }

        while (pathQueue.Count > 0)
        {
            var node = pathQueue.Dequeue();
            var square = node.lastPosition;
            
            if (node.level > perception)
                break;
            if (square.row == goal_row && square.column == goal_col)
                return node.goal;

            if (visited.Contains(node.lastPosition))
                continue;

            visited.Add(node.lastPosition);

            foreach (Direction direction in availableDirections)
            {
                (int row, int col) nextRowCol = GetRowAndColumn(square.row, square.column, direction);
                string cell = Map.GetCell(nextRowCol.row, nextRowCol.col);

                if (visited.Contains(nextRowCol))
                    continue;
                if (CanVisit(cell))
                {
                    var next_node = (node.goal, nextRowCol, node.level + 1);
                    pathQueue.Enqueue(next_node);
                }       
            }
        }
        return (-1, -1);
    }

    public bool CanVisit(string cell)
    {
        switch (cell)
        {
            case ".":
                return true;
        }
        return false;
    }

    public Direction GetDirection(int row, int column, int nextRow, int nextColumn)
    {
        Direction directionHorizontal = Direction.none;
        Direction directionVertical = Direction.none;

        if (row < nextRow)
            directionVertical = Direction.south;
        else if (row > nextRow)
            directionVertical = Direction.north;

        if (column < nextColumn)
            directionHorizontal = Direction.east;
        else if (column > nextColumn)
            directionHorizontal = Direction.west;

        if (directionVertical == Direction.none)
            return directionHorizontal;
        if (directionHorizontal == Direction.none)
            return directionVertical;
        if (directionHorizontal == Direction.east && directionVertical == Direction.south)
            return Direction.southEast;
        if (directionHorizontal == Direction.east && directionVertical == Direction.north)
            return Direction.northEast;
        if (directionHorizontal == Direction.west && directionVertical == Direction.south)
            return Direction.southWest;
        if (directionHorizontal == Direction.west && directionVertical == Direction.north)
            return Direction.nortWest;

        return Direction.none;
    }

    public (int, int) GetRowAndColumn(int row, int column, Direction direction)
    {
        int nextRow = row;
        int nextCol = column;

        switch (direction)
        {
            case Direction.north:
                nextRow -= 1;
                break;
            case Direction.east:
                nextCol += 1;
                break;
            case Direction.west:
                nextCol -= 1;
                break;
            case Direction.south:
                nextRow += 1;
                break;
            case Direction.northEast:
                nextRow -= 1;
                nextCol += 1;
                break;
            case Direction.nortWest:
                nextRow -= 1;
                nextCol -= 1;
                break;
            case Direction.southEast:
                nextRow += 1;
                nextCol += 1;
                break;
            case Direction.southWest:
                nextRow += 1;
                nextCol -= 1;
                break;
            case Direction.player:
                int playerRow = Mathf.RoundToInt(Player.z / Map.scale);
                int playerCol = Mathf.RoundToInt(Player.x / Map.scale);

                if (playerRow > row)
                    nextRow += 1;
                else if (playerRow < row)
                    nextRow -= 1;

                if (playerCol > column)
                    nextCol += 1;
                else if (playerCol < column)
                    nextCol -= 1;

                break;
            default:
                break;
        }
        return (nextRow, nextCol);
    }

    public bool WallCollision(Direction direction)
    {
        Vector3 oldPosition = transform.position;
        Move(direction);

        GameObject hitbox = transform.GetChild(1).gameObject;
        if (Physics.CheckSphere(transform.position, hitbox.GetComponent<CapsuleCollider>().radius, LayerMask.GetMask("Wall")))
        {
            transform.position = oldPosition;
            return true;
        }
        transform.position = oldPosition;
        return false;
    }


    public void Move(Direction direction)
    {
        float y = transform.position.y;
        float x = transform.position.x; 
        float z = transform.position.z; 

        float dz = Mathf.Sin(Mathf.Deg2Rad * 45) * Time.deltaTime;
        float dx = Mathf.Cos(Mathf.Deg2Rad * 45) * Time.deltaTime;

        switch (direction)
        {
            case Direction.north:
                z -= speed * Time.deltaTime;
                break;
            case Direction.east:
                x += speed * Time.deltaTime;
                break;
            case Direction.west:
                x -= speed * Time.deltaTime;
                break;
            case Direction.south:
                z += speed * Time.deltaTime;
                break;
            case Direction.northEast:
                z -= dx;
                x += dz;
                break;
            case Direction.nortWest:
                z -= dx;
                x -= dz;
                break;
            case Direction.southEast:
                z += dx;
                x += dz;
                break;
            case Direction.southWest:
                z += dx;
                x -= dz;
                break;
            case Direction.player:
                Vector3 dv = Vector3.MoveTowards(new Vector3(x, 0, z), new Vector3(Player.x, 0, Player.z), speed);
                z += dv.z * Time.deltaTime;
                x += dv.x * Time.deltaTime;
                break;
            default:
                break;
        }
        transform.position = new Vector3(x, y, z);
    }

    public void SetValues()
    {
        health = data.health;
        damage = data.damage;
        horror = data.horror;
        speed = data.speed;
    }

    public bool OccupiesPlayersSpace()
    {
        int col = Mathf.RoundToInt(transform.position.x / Map.scale);
        int row = Mathf.RoundToInt(transform.position.z / Map.scale);

        int playerCol = Mathf.RoundToInt(Player.x / Map.scale);
        int playerRow = Mathf.RoundToInt(Player.z / Map.scale);

        return (playerCol == col) && (playerRow == row);
    }

    public bool IsCentered()
    {
        int col = Mathf.RoundToInt(transform.position.x / Map.scale);
        int row = Mathf.RoundToInt(transform.position.z / Map.scale);
        //string cell = Map.GetCell(row, col);
        (float x, float z) cellCoords = Map.CellCoordinates(row, col);

        GameObject hitbox = transform.GetChild(1).gameObject;
        float radius = hitbox.GetComponent<CapsuleCollider>().radius;

        float x1 = transform.position.x - radius;
        float x2 = transform.position.x + radius;
        
        float bounding_x1 = cellCoords.x - (Map.scale / 2);
        float bounding_x2 = cellCoords.x + (Map.scale / 2);

        if (x1 < bounding_x1 || x2 > bounding_x2)
            return false;

        float z1 = transform.position.z - radius;
        float z2 = transform.position.z + radius;
        
        float bounding_z1 = cellCoords.z - (Map.scale / 2);
        float bounding_z2 = cellCoords.z + (Map.scale / 2);

        if (bounding_z1 > z1 || bounding_z2 < z2)
            return false;
        
        return true;
    }
}
