using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.ProBuilder.MeshOperations;

public class SequenceCheck : MonoBehaviour
{
    public List<SequenceSocketHelper> sequenceSocketHelpers = new List<SequenceSocketHelper>();
    public Animator rewardAnimator;
    public GameObject sequenceSocketHelpersHolder; //must be set to SelectableSockets without left and right ends
    public GameObject key;
    void Awake()
    {
        if (sequenceSocketHelpersHolder == null)
        {
            Debug.LogError("Gameobject holding sockets for sequence puzzle has not been assigned! ");
            return;
        }

        key.GetComponent<Rigidbody>().isKinematic = true;
         sequenceSocketHelpersHolder.GetComponentsInChildren(sequenceSocketHelpers);
    }

    public void CheckAll()
    {
        foreach (var ch in sequenceSocketHelpers)
        {
            var good= ch.CheckPair();
            if (!good)
            {
                Debug.Log("wrong pairing found for"+ ch.sequenceIdData.name);

                return;
            }  
        }
        Debug.Log("correct pairings found for all");

        Reward();
    }

    public void Reward()
    {
        rewardAnimator.Play("rewardSocket");
    }
}
