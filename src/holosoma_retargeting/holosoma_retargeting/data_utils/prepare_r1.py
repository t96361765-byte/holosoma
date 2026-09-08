"""Package the official R1 URDF for the shared OmniRetarget pipeline.

The source is not edited. Added sole/toe frames have no visual, collision or
inertia. Contact geometry remains the official mesh; the head is locked by
RobotConfig during retargeting, not removed from the model.
"""
from pathlib import Path
import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import trimesh


def prepare(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'meshes').mkdir(exist_ok=True)
    for element in root.findall('.//mesh'):
        src = (source.parent / element.attrib['filename']).resolve()
        dest = output / 'meshes' / src.name
        shutil.copy2(src, dest)
        element.set('filename', 'meshes/' + src.name)

    for side in ('left', 'right'):
        mesh = trimesh.load(source.parent / 'meshes' / f'{side}_ankle_roll_link.STL', force='mesh')
        low, high = mesh.bounds
        # Inset rectangle on the sole, plus a front-foot Laplacian point.
        points = [(x, y, low[2]) for x in (low[0]+0.01, high[0]-0.01)
                  for y in (low[1]+0.008, high[1]-0.008)]
        points.append((high[0]-0.01, (low[1]+high[1])/2, low[2]))
        for i, point in enumerate(points, 1):
            name = f'{side}_sole_{i}_link' if i <= 4 else f'{side}_toe_link'
            ET.SubElement(root, 'link', name=name)
            joint = ET.SubElement(root, 'joint', name=name+'_fixed', type='fixed')
            ET.SubElement(joint, 'parent', link=f'{side}_ankle_roll_link')
            ET.SubElement(joint, 'child', link=name)
            ET.SubElement(joint, 'origin', xyz=' '.join(f'{v:.12g}' for v in point), rpy='0 0 0')

    urdf = output / 'r1_26dof.urdf'
    ET.indent(tree, space='  ')
    tree.write(urdf, encoding='utf-8', xml_declaration=True)
    # Import all visuals and retain fixed semantic frames in the compiled MJCF.
    mj = root.find('mujoco')
    if mj is None:
        mj = ET.SubElement(root, 'mujoco')
    compiler = mj.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(mj, 'compiler')
    compiler.attrib.update(discardvisual='false', fusestatic='false', balanceinertia='true', strippath='false')
    compiler.attrib.pop('meshdir', None)
    for element in root.findall('.//mesh'):
        element.set('filename', str((output / element.attrib['filename']).resolve()))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tree.write(tmp / 'import.urdf', encoding='utf-8', xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(tmp / 'import.urdf'))
        mujoco.mj_saveLastXML(str(tmp / 'import.xml'), model)
        result = ET.parse(tmp / 'import.xml')
    xml = result.getroot()
    compiler = xml.find('compiler')
    compiler.attrib.pop('meshdir', None)
    for mesh in xml.findall('./asset/mesh'):
        mesh.set('file', 'meshes/' + Path(mesh.attrib['file']).name)
    body = xml.find('./worldbody/body')
    ET.SubElement(body, 'freejoint', name='floating_base')
    ET.SubElement(xml.find('worldbody'), 'geom', name='ground', type='plane', size='10 10 0.1')
    ET.indent(result, space='  ')
    xml_path = output / 'r1_26dof.xml'
    result.write(xml_path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    assert model.nq == 33 and model.nv == 32, (model.nq, model.nv)
    assert names[24:] == ['head_pitch_joint', 'head_yaw_joint'], names
    assert all(model.body(f'{s}_toe_link').id >= 0 for s in ('left', 'right'))
    print(f'Prepared {xml_path}: nq={model.nq}, nv={model.nv}, 26 joints; head indices 24/25')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'models' / 'r1')
    args = parser.parse_args()
    prepare(args.source, args.output)
