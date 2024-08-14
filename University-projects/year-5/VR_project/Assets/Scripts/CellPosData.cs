using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;


[CreateAssetMenu(menuName = "Cell Data")]
public class CellPosData : ScriptableObject
{
    [SerializeField] List<DataHolder> list;

    [System.Serializable]
    public class DataHolder
    {
        [SerializeField] public List<Transform> startPos;

        [SerializeField] public List<Transform> goalPosm;
    }
}