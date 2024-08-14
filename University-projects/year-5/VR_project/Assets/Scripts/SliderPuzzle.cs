using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

//we use acompanying createCellStartGoalPositions.cs editor script to set goal, start vec3 positions 
public class SliderPuzzle : MonoBehaviour
{
    //sliced images to be assigned to the cell 
    public List<Sprite> sprites = new List<Sprite>();
    //the cell behaviour script
    public List<SlidePuzzleCell> slidePuzzleCells = new List<SlidePuzzleCell>();

    //bool to save if the puzzle is solved correctly- need to set false if we return big distance
    public bool correct = false;
    //distance,bias that the cell is alowed to deviate from perfect position  
    public float maxDistance = 0.5f;
    public GameObject keyReward;

    //object that holds cell objects
    public GameObject sliderCellsHolder;

    //list that holds renderers to be assigned sprites
    public SpriteRenderer[] spriteRenderers;


    //get access to transofrms of the cels to scramble them and set starting/goal pos
    //references to the objects
    [SerializeField] public List<Transform> transforms;
    //list of scrambled positons
    [SerializeField] public List<Vector3> startPos;
    // list of correct positions
    [SerializeField] public List<Vector3> goalPos;


    //allow some small distance to the perfect goal location
    void Start()
    {
        //assign them to list instantly
        sliderCellsHolder.GetComponentsInChildren(slidePuzzleCells);

        spriteRenderers = sliderCellsHolder.GetComponentsInChildren<SpriteRenderer>();
        for(int i = 0; i<spriteRenderers.Length; i++)
        {
            spriteRenderers[i].sprite = sprites[i];
        }

        //ShufflePuzzle(); dont bother
    }
    
    //function to generate scrambled puzzle-start from end and move randomly if we can
    void ShufflePuzzle()
    {
        for(int i = 0; i<spriteRenderers.Length; i++)
        {
            //spriteRenderers[i].sprite = sprites[i];
            //slidePuzzleCells[i].transform.position = ;
        }
    }

    //called from the cell after we deselect the cell
    public void CheckCorrectness()
    {
        for(int i = 0; i<spriteRenderers.Length; i++)
        {
            var dist = Vector3.Distance(slidePuzzleCells[i].transform.localPosition, goalPos[i]);
            if (dist <= maxDistance)
            {
                Debug.Log("distance is, "+ dist);
  
            }
            else
            {
                Debug.Log("distance was too big, "+ dist);
                return;
            }

            if (i == spriteRenderers.Length - 1)
            {
                //only run this if we find all to be close enough
                Debug.Log("CONGRATS, you solved the puzzle");
                PuzzleReward();
            }

        }
    }
    
    // use this function to reward thr player for completing the puzzle
    public void PuzzleReward()
    {
        //drop key
        //make this simple for now
        keyReward.SetActive(true);
        keyReward.transform.SetParent(null);
        keyReward.GetComponentInChildren<Rigidbody>().isKinematic = false;
    }
}
