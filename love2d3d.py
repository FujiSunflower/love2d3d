#
# Copyright 2018 Fuji Sunflower
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bpy
import numpy as np
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from bpy_extras.view3d_utils import location_3d_to_region_2d
import bgl
import blf
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
#from mathutils.bvhtree import BVHTree
import sys
#import datetime

bl_info = {
    "name": "Love2D3D",
    "author": "Fuji Sunflower",
    "version": (2, 0),
    "blender": (2, 79, 0),
    "location": "3D View > Object Mode > Tool Shelf > Create > Love2D3D",
    "description": "Create 3D object from 2D image",
    "warning": "",
    "support": "COMMUNITY",
    "wiki_url": "https://github.com/FujiSunflower/love2d3d/wiki",
    "tracker_url": "https://github.com/FujiSunflower/love2d3d/issues",
    "category": "Add Mesh"
}

RGBA = 4  # Color size per pixels
RGB = 3  # Color size per pixels
R = 0  # Index of color
G = 1  # Index of color
B = 2  # Index of color
A = 3  # Index of color
X = 0  # Index
Y = 1  # Index
LEFT = 2  # Index
RIGHT = 3  # Index
BOTTOM = 4  # Index
TOP = 5  # Index
QUAD = 4  # Vertex Numer of Quad
FRONT = 0
BACK = 1
NAME = "Love2D3D"  # Name of 3D object
BOUND_LEFT = 0
BOUND_RIGHT = 1
BOUND_BACK = 2
BOUND_FRONT = 3
BOUND_TOP = 4
BOUND_BOTTOM = 5
BOUND_CENTER = 6
#LATTICE_RESOLUTION = 6.0
BRANCH_BOOST = 3.0
#BRANCH_DISPERSION_RATIO = 0.1
#BRANCH_LIMIT_HIPS = np.radians(5.0) 
#BRANCH_LIMIT_CENTER = np.radians(5.0) 
#BRANCH_LIMIT_ARM = np.radians(30.0)
#BRANCH_LIMIT_LEG = np.radians(5.0)
#BRANCH_LIMIT_ANY = np.radians(30.0)
#BRANCH_LIMIT_FINGER = np.radians(3.0)
BONE_TYPE_ANY = -1
BONE_TYPE_BODY = 0
BONE_TYPE_HEAD = 1
BONE_TYPE_ARM_LEFT = 2
BONE_TYPE_ARM_RIGHT = 3
BONE_TYPE_LEG_LEFT = 4
BONE_TYPE_LEG_RIGHT = 5
#BONE_TYPE_FINGER = 6
BONE_TYPE_FINGER_LEFT = 7
BONE_TYPE_FINGER_RIGHT = 8

def draw_callback_px(self, context):
    #print("mouse points", len(self.mouse_path))

    #font_id = 0  # XXX, need to find out how best to get this.

    # draw some text
    #blf.position(font_id, 15, 30, 0)
    #blf.size(font_id, 20, 72)
    #blf.draw(font_id, "Hello Word " + str(len(self.mouse_path)))

    image = context.window_manager.love2d3d.image_front  # Image ID
    if image == "" or image is None:
        return
    self.image = context.blend_data.images[image]  # Get image

    # 50% alpha, 2 pixel width line
    #bgl.glLineWidth(0)
    bgl.glEnable(bgl.GL_BLEND)
    #bgl.glClearColor(0.0, 0.0, 0.0, 0.0);
    #bgl.glBlendFunc(bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA)
    self.image.gl_load()
    bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.image.bindcode[0])
    bgl.glColor4f(1.0, 1.0, 1.0, 0.5)
    bgl.glEnable(bgl.GL_TEXTURE_2D)
    bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_NEAREST)
    bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_NEAREST)
    #bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
    #bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)

    #self.image.bind()
    #bgl.glLineWidth(2)

#    bgl.glBegin(bgl.GL_LINE_STRIP)
#    for x, y in self.mouse_path:
#        bgl.glVertex2i(x, y)
#
#    bgl.glEnd()
    #camera = context.space_data.camera.location
    #sclip = context.space_data.clip_start
    #eclip = context.space_data.clip_end
    #lens = context.space_data.lens
    #pers = context.region_data.perspective_matrix
    #fovy = math.atan(pers[5]) * 2
    #aspect = pers[5] / pers[0]
    #bgl.gluPerspective(fovy, aspect, sclip, eclip);
    bgl.glMatrixMode(bgl.GL_MODELVIEW)
    #print(context.region_data.view_matrix)
    #ob = context.active_object
    #buff = bgl.Buffer(bgl.GL_FLOAT, [4, 4], context.region_data.view_matrix.transposed())
    #buff = bgl.Buffer(bgl.GL_FLOAT, [4, 4], ob.matrix_world.transposed())
    #mat = Matrix.Identity(4)
    mat = Matrix.Translation(context.space_data.cursor_location)
    view_align = context.window_manager.love2d3d.view_align
    if view_align:
        iview = Matrix(context.region_data.view_matrix).inverted_safe().to_3x3().to_4x4()
    else:
        iview = Matrix.Identity(4)
    buff = bgl.Buffer(bgl.GL_FLOAT, [4, 4], (mat * iview).transposed())
    bgl.glLoadMatrixf(buff)
    #bgl.glLoadIdentity()
    #camera = context.region_data.view_location
    #bgl.gluLookAt(camera.x, camera.y, camera.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    bgl.glMatrixMode(bgl.GL_PROJECTION);
    #bgl.glLoadIdentity();
    buff = bgl.Buffer(bgl.GL_FLOAT, [4, 4], context.region_data.perspective_matrix.transposed())
    bgl.glLoadMatrixf(buff)

    scale = context.window_manager.love2d3d.scale
    w, h = self.image.size
    w *= scale
    h *= scale
    #lb = location_3d_to_region_2d(context.region, context.space_data.region_3d, (-w / 2.0, 0, -h / 2.0))
    #rb = location_3d_to_region_2d(context.region, context.space_data.region_3d, (w / 2.0, 0, -h / 2.0))
    #rt = location_3d_to_region_2d(context.region, context.space_data.region_3d, (w / 2.0, 0, h / 2.0))
    #lt = location_3d_to_region_2d(context.region, context.space_data.region_3d, (-w / 2.0, 0, h /2.0))
    if view_align:
        lb = Vector((-w / 2.0, -h / 2.0, 0))
        rb = Vector((w / 2.0, -h / 2.0, 0))
        rt = Vector((w / 2.0, h / 2.0, 0))
        lt = Vector((-w / 2.0, h /2.0, 0))
    else:
        lb = Vector((-w / 2.0, 0, -h / 2.0))
        rb = Vector((w / 2.0, 0, -h / 2.0))
        rt = Vector((w / 2.0, 0, h / 2.0))
        lt = Vector((-w / 2.0, 0, h /2.0))
    #print(lt)
    bgl.glBegin(bgl.GL_QUADS)
    bgl.glTexCoord2d(0.0, 0.0)
    bgl.glVertex3f(lb.x, lb.y, lb.z)
    bgl.glTexCoord2d(1.0, 0.0)
    bgl.glVertex3f(rb.x, rb.y, lb.z)
    bgl.glTexCoord2d(1.0, 1.0)
    bgl.glVertex3f(rt.x, rt.y, rt.z)
    bgl.glTexCoord2d(0.0, 1.0)
    bgl.glVertex3f(lt.x, lt.y, lt.z)
    bgl.glEnd()
    self.image.gl_free()
    # restore opengl defaults
    bgl.glLineWidth(1)
    bgl.glDisable(bgl.GL_TEXTURE_2D)
    bgl.glDisable(bgl.GL_BLEND)
    bgl.glColor4f(0.0, 0.0, 0.0, 1.0)


class Preview(bpy.types.Operator):

    bl_idname = "object.preview_love2d3d"
    bl_label = "Preview love2D3D"
    bl_description = "Preview love2D3D"
    bl_options = {'INTERNAL'}
    _handle = None

    def modal(self, context, event):
        area = context.area        
        if area is None:
            return {'PASS_THROUGH'}
        area.tag_redraw()
        preview = context.window_manager.love2d3d.preview
        if not preview:
            return {'FINISHED'}
        #if event.type == 'MOUSEMOVE':
        #    #self.mouse_path.append((event.mouse_region_x, event.mouse_region_y))
        #    pass
        #elif event.type == 'LEFTMOUSE':
        #    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        #    #self.image.gl_free()
        #    return {'FINISHED'}
        #
        #elif event.type in {'RIGHTMOUSE', 'ESC'}:
        #    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        #    #self.image.gl_free()
        #    return {'CANCELLED'}

        #image = context.window_manager.love2d3d.image_front  # Image ID
        #if image == "":
        #    #return {"CANCELLED"}
        #    self.image = context.blend_data.images[image]  # Get image

        #return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}
    def _handle_remove(self, context):
        if Preview._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                                                      Preview._handle, 'WINDOW')
            Preview._handle = None
    def _handle_add(self, context):
        if Preview._handle is None:
            Preview._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
    
    def invoke(self, context, event):
        preview = context.window_manager.love2d3d.preview
        if context.area.type == 'VIEW_3D':
            # the arguments we pass the the callback
            args = (self, context)
            # Add the region OpenGL drawing callback
            # draw in view space with 'POST_VIEW' and 'PRE_VIEW'
            if not preview:
                context.window_manager.love2d3d.preview = True
            else:
                context.window_manager.love2d3d.preview = False
                self._handle_remove(context)
                return {'FINISHED'}
#            self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
            self._handle_add(context)
            #image = context.window_manager.love2d3d.image_front  # Image ID
            #if image == "":
            #    return {"CANCELLED"}
            #self.image = context.blend_data.images[image]  # Get image
            #self.image.gl_load()
            #self.mouse_path = []

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            self._handle_remove(context)
            return {'CANCELLED'}


class CreateObject(bpy.types.Operator, AddObjectHelper):

    bl_idname = "object.create_love2d3d"
    bl_label = "Create love2D3D"
    bl_description = "Create 3D object from 2D image"
    bl_options = {'REGISTER', 'UNDO'}
    #view_align = bpy.context.window_manager.love2d3d.view_align
    #view_align = bpy.props.BoolProperty(name="View align",
    #                                 description="Use view align for mesh")
    #view_align = True

    def execute(self, context):
        #debug_time = datetime.datetime.today()
        image = context.window_manager.love2d3d.image_front  # Image ID
        if image == "":
            return {"CANCELLED"}
        image = context.blend_data.images[image]  # Get image
        resolution = context.window_manager.love2d3d.rough  # Get resolution
        w, h = image.size  # Image width and height
        all = w * h
        pixels = image.pixels[:]  # Get slice of color infomation
        fronts = []
        backs = [[True for i in range(w)] for j in range(h)] # Whether background or not
        ex = h - resolution # End of list
        ey = w - resolution # End of list
        opacity = context.window_manager.love2d3d.opacity # Whether use opacity or not
        threshold = context.window_manager.love2d3d.threshold # threshold of background
        for y in range(resolution, ex)[::resolution]:
            left = 0 + y * w
            il = RGBA * left  # Get left index of color in image
            for x in range(resolution, ey)[::resolution]:
                back = False
                for v in range(resolution):
                    for u in range(resolution):
                        p = (x + u) + (y + v) * w #cuurent index in pixels
                        i = RGBA * p  #Get each index of color in image
                        if opacity:  # Whether opaque or not
                            c = pixels[i + A] # each opacity in image
                            cl = pixels[il + A] # left opacity in image
                            back = back or c <= threshold
                        else:  # Whether same color or not
                            c = pixels[i:i + RGB] # each RGB in image
                            cl = pixels[il:il + RGB] # left RGB in image
                            back = back or abs(c[R] - cl[R]) + \
                                abs(c[G] - cl[G]) \
                                + abs(cl[B] - cl[B]) <= threshold * 3.0
                        if back:
                            break
                    if back:
                        break
                backs[y][x] = back
                if not back:
                    fronts.append((x // resolution, y // resolution))
        del ex, ey, i, il, c, cl, back, pixels, p, left
        terms = [] #Edges of image
        for k, f in enumerate(fronts):
            fx = f[X]
            fy = f[Y]
            x = fx * resolution
            y = fy * resolution
            left = backs[y][x-resolution]
            right = backs[y][x+resolution]
            back = backs[y-resolution][x]
            top = backs[y+resolution][x]
            if not backs[y][x] and (left or right or back or top):
                terms.append((fx, fy))  # Get edge
            fronts[k] = (fx, fy, left, right, back, top)  # Insert edge info
        lens = [[0.0 for i in range(w)[::resolution]]
                for j in range(h)[::resolution]]
        if len(fronts) == 0:
            return {"CANCELLED"}
        sqAll = all ** 2
        
        #xs = np.array([f[X] for f in fronts])  # X coordinates of each point
        #ys = np.array([f[Y] for f in fronts])  # Y coordinates of each point
        #ls = np.full(len(fronts), sqAll)
        #for t in terms:
        #    ms = np.minimum(ls, np.power(t[X] - xs, 2) + np.power(t[Y] - ys, 2))
        #    ls = ms  # Watershed algorithm        
        kd = KDTree(len(terms))
        for i, t in enumerate(terms):
            kd.insert((t[X], t[Y], 0), i)
        kd.balance()
        ls = [0.0 for f in fronts]
        for k, f in enumerate(fronts):
            co_find = (f[X], f[Y], 0)
            co, index, dist = kd.find(co_find)
            ls[k] = dist
        #ms = np.sqrt(ls) + 1 # length array with softning
        ms = np.array([l + 1 for l in ls])
        m = np.max(ms)
        ls = np.divide(ms, m)  # Nomalize
        ms = (np.sin(ls * np.pi * 0.5) + 0)
        #ms = (np.arcsin(ls) + 0)

        for k, f in enumerate(fronts):
            fx = f[X]
            fy = f[Y]
            ls = ms[k] / 4.0  # Blur of height for edge
            lens[fy][fx] += ls
            fxi = fx + 1
            fyi = fy + 1
            lens[fy][fxi] += ls
            lens[fyi][fx] += ls
            lens[fyi][fxi] += ls
        del fx, fy, fxi, fyi, left, right, back, top, k, f, ms, ls, m
        verts = []
        nei = 1  # Neighbor
        uvs = []
        uvx = 0 / w
        uvy = 0 / h
        backs = []
        #self.view_align = context.window_manager.love2d3d.view_align
        view_align = context.window_manager.love2d3d.view_align
        scale = context.window_manager.love2d3d.scale
        s = min(w, h) / 8
        depth_front = s * context.window_manager.love2d3d.depth_front * scale
        depth_back = s * context.window_manager.love2d3d.depth_back * scale
        for f in fronts:
            x = f[X]
            y = f[Y]
            xi = x + nei
            yi = y + nei
            x1 = x * resolution
            x2 = xi * resolution
            y1 = y * resolution
            y2 = yi * resolution
            lu = x1 / w
            ru = x2 / w
            bu = y1 / h
            tu = y2 / h
            x1 = (x1 - w / 2) * scale
            x2 = (x2 - w / 2) * scale
            y1 = (y1 - h / 2) * scale
            y2 = (y2 - h / 2) * scale

            # Front face

            #p1 = (x1, -lens[yi][x] * depth_front, y2)
            #p2 = (x1, -lens[y][x] * depth_front, y1)
            #p3 = (x2, -lens[y][xi] * depth_front, y1)
            #p4 = (x2, -lens[yi][xi] * depth_front, y2)
            if view_align:
                p1 = (x1, y2, lens[yi][x] * depth_front)
                p2 = (x1, y1, lens[y][x] * depth_front)
                p3 = (x2, y1, lens[y][xi] * depth_front)
                p4 = (x2, y2, lens[yi][xi] * depth_front)
            else:
                p1 = (x1, -lens[yi][x] * depth_front, y2)
                p2 = (x1, -lens[y][x] * depth_front, y1)
                p3 = (x2, -lens[y][xi] * depth_front, y1)
                p4 = (x2, -lens[yi][xi] * depth_front, y2)
            verts.extend([p1, p2, p3, p4])
            u1 = (lu + uvx, tu + uvy)
            u2 = (lu + uvx, bu + uvy)
            u3 = (ru + uvx, bu + uvy)
            u4 = (ru + uvx, tu + uvy)
            uvs.extend([u1, u2, u3, u4])
            backs.append(FRONT)
            # Back face
            if view_align:
                p5 = (x2,  y2, -lens[yi][xi] * depth_back)
                p6 = (x2, y1, -lens[y][xi] * depth_back)
                p7 = (x1, y1, -lens[y][x] * depth_back)
                p8 = (x1, y2, -lens[yi][x] * depth_back)
            else:
                p5 = (x2, lens[yi][xi] * depth_back, y2)
                p6 = (x2, lens[y][xi] * depth_back, y1)
                p7 = (x1, lens[y][x] * depth_back, y1)
                p8 = (x1, lens[yi][x] * depth_back, y2)
            verts.extend([p5, p6, p7, p8])
            uvs.extend([u4, u3, u2, u1])
            backs.append(BACK)
            if f[LEFT]:  # Left face
                verts.extend([p8, p7, p2, p1])
                uvs.extend([u1, u2, u2, u1])
                backs.append(FRONT)
            if f[RIGHT]:  # Right face
                verts.extend([p4, p3, p6, p5])
                uvs.extend([u4, u3, u3, u4])
                backs.append(FRONT)
            if f[TOP]:  # Top face
                verts.extend([p8, p1, p4, p5])
                uvs.extend([u1, u1, u4, u4])
                backs.append(FRONT)
            if f[BOTTOM]:  # Bottom face
                verts.extend([p2, p7, p6, p3])
                uvs.extend([u2, u2, u3, u3])
                backs.append(FRONT)
        del p1, p2, p3, p4, p5, p6, p7, p8, lens, nei, x, y
        del xi, yi, lu, ru, bu, tu, x1, x2, y1, y2
        del u1, u2, u3, u4
        faces = [(0, 0, 0, 0)] * (len(verts) // QUAD)
        for n, f in enumerate(faces):
            faces[n] = (QUAD * n, QUAD * n + 1, QUAD * n + 2, QUAD * n + 3)
        msh = bpy.data.meshes.new(NAME)
        msh.from_pydata(verts, [], faces)  # Coordinate is Blender Coordinate
        msh.update()
        del verts, faces
        obj = object_data_add(context, msh, operator=self).object
        if view_align:
            iview = Matrix(context.region_data.view_matrix).inverted_safe().to_quaternion()
            angle = iview.angle
            axis = iview.axis
            bpy.ops.transform.rotate(value=angle, axis=axis)
        #obj = object_data_add(context, msh).object
        #obj = bpy.data.objects.new(NAME, msh)  # Create 3D object
        #scene = bpy.context.scene
        #scene.objects.link(obj)
        #bpy.ops.object.select_all(action='DESELECT')
        #if bpy.ops.object.mode_set.poll():
        #    bpy.ops.object.mode_set(mode='OBJECT')
        #obj.select = True
        context.scene.objects.active = obj
#        obj.location = (-w/2, 0, -h/2)  # Translate to origin
#        bpy.ops.object.transform_apply(location=True)
#        scale = context.window_manager.love2d3d.scale
#        obj.scale = (scale, scale, scale)
#        bpy.ops.object.transform_apply(scale=True)
#        obj.location = context.space_data.cursor_location

        channel_name = "uv"
        msh.uv_textures.new(channel_name)  # Create UV coordinate
        for idx, dat in enumerate(msh.uv_layers[channel_name].data):
            dat.uv = uvs[idx]
        del uvs
        # Crate fornt material
        matf = bpy.data.materials.new('Front')
        tex = bpy.data.textures.new('Front', type='IMAGE')
        tex.image = image
        matf.texture_slots.add()
        matf.texture_slots[0].texture = tex
        matf.use_shadeless = context.window_manager.love2d3d.shadeless
        obj.data.materials.append(matf)
        # Crate back material
        matb = bpy.data.materials.new('Back')
        tex = bpy.data.textures.new('Back', type='IMAGE')
        image_back = context.window_manager.love2d3d.image_back
        if image_back == "":
            tex.image = image
        else:
            image_back = context.blend_data.images[image_back]
            tex.image = image_back
        matb.texture_slots.add()
        matb.texture_slots[0].texture = tex
        matb.use_shadeless = context.window_manager.love2d3d.shadeless
        obj.data.materials.append(matb)
        for k, f in enumerate(obj.data.polygons):
            f.material_index = backs[k]  # Set back material
        context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')  # Remove doubled point
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene.objects.active = obj  # Apply modifiers
        bpy.ops.object.modifier_add(type='SMOOTH')
        smo = obj.modifiers["Smooth"]
        smo.iterations = context.window_manager.love2d3d.smooth
        bpy.ops.object.modifier_add(type='DISPLACE')
        dis = obj.modifiers["Displace"]
        dis.strength = context.window_manager.love2d3d.fat * scale / 0.01
        dec = None
        if context.window_manager.love2d3d.decimate:
            bpy.ops.object.modifier_add(type='DECIMATE')
            dec = obj.modifiers["Decimate"]
            dec.ratio = context.window_manager.love2d3d.decimate_ratio
        if context.window_manager.love2d3d.modifier:
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Smooth")
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Displace")
            if context.window_manager.love2d3d.decimate:
                bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")
        obj.select = True
        bpy.ops.object.shade_smooth()
        #print(datetime.datetime.today() - debug_time)
        return {'FINISHED'}
    def draw(self, context):
        layout = self.layout
        #col = layout.column()
        #col.label(text="Custom Interface!")
        #row = col.row()
        #row.prop(self, "my_float")
        #row.prop(self, "my_bool")
        #layout.prop(self, "my_string")
        #layout.prop(context.window_manager.love2d3d, "view_align")
        #col = layout.column(align=True)
        #col.label(text="Object", icon="OBJECT_DATA")
        #col.operator(CreateObject.bl_idname, text="Create")
        #row = col.row()
        #row.label(text="Preview")
        #preview = context.window_manager.love2d3d.preview
        #row.operator(Preview.bl_idname, text="On" if preview else "Off")
        col = layout.column(align=True)
        col.label(text="Image", icon="IMAGE_DATA")
        #col.operator("image.open", icon="FILESEL")
        col.prop_search(context.window_manager.love2d3d,
                        "image_front", context.blend_data, "images")
        col.prop_search(context.window_manager.love2d3d,
                        "image_back", context.blend_data, "images")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Separation", icon="IMAGE_RGB_ALPHA")
        col.prop(context.window_manager.love2d3d, "threshold")
        col.prop(context.window_manager.love2d3d, "opacity")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Geometry", icon="EDITMODE_HLT")
        col.prop(context.window_manager.love2d3d, "view_align")
        col.prop(context.window_manager.love2d3d, "depth_front")
        col.prop(context.window_manager.love2d3d, "depth_back")
        col.prop(context.window_manager.love2d3d, "scale")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Quality", icon="MOD_SMOOTH")
        col.prop(context.window_manager.love2d3d, "rough")
        col.prop(context.window_manager.love2d3d, "smooth")
        col.prop(context.window_manager.love2d3d, "fat")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Decimate", icon="MOD_DECIM")        
        col.prop(context.window_manager.love2d3d, "decimate")
        col.prop(context.window_manager.love2d3d, "decimate_ratio")
        layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Armature", icon="ARMATURE_DATA")        
        #col.operator(CreateArmature.bl_idname, text="Create")
        #col.prop(context.window_manager.love2d3d, "armature_resolution")
        #col.prop(context.window_manager.love2d3d, "armature_finger_resolution")
        #layout.separator()
        col = layout.column(align=True)
        col.label(text="Option", icon="SCRIPTWIN")
        col.prop(context.window_manager.love2d3d, "modifier")
        col.prop(context.window_manager.love2d3d, "shadeless")

class CreateArmature(bpy.types.Operator):

    bl_idname = "object.create_love2d3d_aramature"
    bl_label = "Create Armature"
    bl_description = "Create Armature to selected objects"
    bl_options = {'REGISTER', 'UNDO'}
    finger_limit_angle = bpy.props.FloatProperty(name="Finger limit",
                                          description="Limit angle of finger",
                                          min=0.0, default=np.radians(8.0), subtype='ANGLE')
    hips_limit_angle = bpy.props.FloatProperty(name="Hips limit",
                                          description="Limit angle of hips",
                                          min=0.0, max=np.radians(90.0), default=np.radians(5.0), subtype='ANGLE')
    center_limit_angle = bpy.props.FloatProperty(name="Center limit",
                                          description="Limit angle of center",
                                          min=0.0, max=np.radians(90.0), default=np.radians(5.0), subtype='ANGLE')
    arm_limit_angle = bpy.props.FloatProperty(name="Arm limit",
                                          description="Limit angle of arm",
                                          min=0.0, max=np.radians(90.0), default=np.radians(30.0), subtype='ANGLE')
    leg_limit_angle = bpy.props.FloatProperty(name="Leg limit",
                                          description="Limit angle of leg",
                                          min=0.0, max=np.radians(90.0), default=np.radians(5.0), subtype='ANGLE')
    any_limit_angle = bpy.props.FloatProperty(name="Any limit",
                                          description="Limit angle of any bone",
                                          min=0.0, max=np.radians(90.0), default=np.radians(30.0), subtype='ANGLE')
    hand_limit_angle = bpy.props.FloatProperty(name="Hand limit",
                                          description="Limit angle of hand",
                                          min=0.0, max=np.radians(90.0), default=np.radians(45.0), subtype='ANGLE')
    branch_boost = bpy.props.FloatProperty(name="Boost",
                                          description="How many points hit as branch",
                                          min=0.01, default=3.0)
    finger_branch_boost = bpy.props.FloatProperty(name="Finger boost",
                                          description="How many points hit as branch in finger",
                                          min=0.01, default=3.0)
    gather_ratio = bpy.props.FloatProperty(name="Gather",
                                          description="How many branchs gather",
                                          min=0.0, max=100.0, default=10, subtype='PERCENTAGE')
    finger_gather_ratio = bpy.props.FloatProperty(name="Finger gather",
                                          description="How many branchs gather in finger",
                                          min=0.0, max=100.0, default=1, subtype='PERCENTAGE')    
    tip_gather_ratio = bpy.props.FloatProperty(name="Tip gather",
                                          description="How many tips gather in finger",
                                          min=0.0, max=100.0, default=3, subtype='PERCENTAGE')    
    #use_finger = bpy.props.BoolProperty(name="Finger",
    #                                  description="Use finger",
    #                                  default=False)

    #BOUND_LEFT = 0
    #BOUND_RIGHT = 1
    #BOUND_BACK = 2
    #BOUND_FRONT = 3
    #BOUND_TOP = 4
    #BOUND_BOTTOM = 5
    #BOUND_CENTER = 6
    ##LATTICE_RESOLUTION = 6.0
    #BRANCH_BOOST = 3.0
    #BRANCH_DISPERSION_RATIO = 0.1
    #BRANCH_LIMIT_HIPS = np.radians(5.0) 
    #BRANCH_LIMIT_CENTER = np.radians(5.0) 
    #BRANCH_LIMIT_ARM = np.radians(30.0)
    #BRANCH_LIMIT_LEG = np.radians(5.0)
    #BRANCH_LIMIT_ANY = np.radians(30.0)
    #BONE_TYPE_ANY = -1
    #BONE_TYPE_BODY = 0
    #BONE_TYPE_HEAD = 1
    #BONE_TYPE_ARM_LEFT = 2
    #BONE_TYPE_ARM_RIGHT = 3
    #BONE_TYPE_LEG_LEFT = 4
    #BONE_TYPE_LEG_RIGHT = 5
    
    def execute(self, context):
        return self.skinning(context)

    def draw(self, context):
        #super.draw(context)
        layout = self.layout
        layout.label(text="Main", icon="ARMATURE_DATA")
        col = layout.column(align=True)
        col.label(text="Resolution", icon="LATTICE_DATA")
        col.prop(context.window_manager.love2d3d, "armature_resolution")
        col = layout.column(align=True)
        col.label(text="Limit", icon="CONSTRAINT")
        col.prop(self, "hips_limit_angle")
        col.prop(self, "center_limit_angle")
        col.prop(self, "arm_limit_angle")
        col.prop(self, "leg_limit_angle")
        col.prop(self, "any_limit_angle")
        col = layout.column(align=True)
        col.label(text="Amount", icon="EDITMODE_HLT")
        col.prop(self, "branch_boost")
        col.prop(self, "gather_ratio")
        layout.separator()
        #layout = layout.column(align=True)
        layout.label(text="Finger", icon="HAND")
        col = layout.column(align=True)
        col.prop(context.window_manager.love2d3d, "armature_finger")
        col.label(text="Resolution", icon="LATTICE_DATA")
        col.prop(context.window_manager.love2d3d, "armature_finger_resolution")
        col = layout.column(align=True)
        col.label(text="Limit", icon="CONSTRAINT")
        col.prop(self, "hand_limit_angle")
        col.prop(self, "finger_limit_angle")
        col = layout.column(align=True)
        col.label(text="Amount", icon="EDITMODE_HLT")
        col.prop(self, "finger_branch_boost")
        col.prop(self, "finger_gather_ratio")
        col.prop(self, "tip_gather_ratio")

    def bound_loc(self, obj):
        """
            Getting bounds of object.
        """
        bound = obj.bound_box
        mat = Matrix(obj.matrix_world)    
        xs = []
        ys = []
        zs = []
        for b in bound:
            loc = mat * Vector(b)
            xs.append(loc.x)
            ys.append(loc.y)
            zs.append(loc.z)
        left = max(xs)
        right = min(xs)
        back = max(ys)
        front = min(ys)
        top = max(zs)
        bottom = min(zs)
        center = Vector(((left + right) * 0.5, (back + front) * 0.5, (top  + bottom) * 0.5))
        return (left, right, back, front, top, bottom, center)
    def primary_obj(self, group):
        """
            Deciding of primary bone in group.
        """  
        max_volume = 0.0
        max_obj = None
        for obj in group:
            b = self.bound_loc(obj)
            le = b[BOUND_LEFT]
            ri = b[BOUND_RIGHT]
            ba = b[BOUND_BACK]
            fr = b[BOUND_FRONT]
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]
            volume = (le - ri) * (ba - fr) * (to - bo)
            if max_volume < volume:
                max_volume = volume
                max_obj = obj
        return max_obj

    def _make_group(self, objects, index, hits):
        """
            Recursion call of objects collision.
        """    
        current_count = len(hits)
        b = self.bound_loc(objects[index])
        le = b[BOUND_LEFT]
        ri = b[BOUND_RIGHT]
        ba = b[BOUND_BACK]
        fr = b[BOUND_FRONT]
        to = b[BOUND_TOP]
        bo = b[BOUND_BOTTOM]
        ce = b[BOUND_CENTER]
        neighbors = []
        for k, neighbor in enumerate(objects):
            if index == k:
                continue
            n = self.bound_loc(neighbor)
            n_le = n[BOUND_LEFT]
            n_ri = n[BOUND_RIGHT]
            n_ba = n[BOUND_BACK]
            n_fr = n[BOUND_FRONT]
            n_to = n[BOUND_TOP]
            n_bo = n[BOUND_BOTTOM]
            n_ce = n[BOUND_CENTER]
            avoid_x = le < n_ri or n_le < ri
            avoid_y = ba < n_fr or n_ba < fr
            avoid_z = to < n_bo or n_to < bo
            avoid = avoid_x or avoid_y or avoid_z
            if not avoid: # Hit
                neighbors.append(k)
        new_hits = []
        for neighbor in neighbors:
            already = False
            for hit in hits:
                already = already or neighbor == hit
            if not already:
                hits.append(neighbor)
        if current_count == len(hits):
            return True
        for h in hits:
           g = self._make_group(objects, h, hits)
           if g:
               return True
           
    def make_group(self, objects):
        """
            Grouping of objects.
        """
        groups = []
        alredys = [False for l in objects]
        for k, object in enumerate(objects):
            if alredys[k]:
                continue
            hits = [k,]
            self._make_group(objects, k, hits)
            group = []
            for hit in hits:
                alredys[hit] = True
                group.append(objects[hit])
            groups.append(group)
        return groups
    def skinning(self, context):
        #debug_time = datetime.datetime.today()
        if len(context.selected_objects) == 0:
            return {"CANCELLED"}
        objects = [] # Only Mesh
        for obj in context.selected_objects:
            if isinstance(obj.data, bpy.types.Mesh):
               objects.append(obj)
        if len(objects) == 0:
            return {"CANCELLED"}
        center = Vector((0, 0, 0))
        sample = 0
        top = -sys.float_info.max
        bottom = sys.float_info.max
    
        for obj in objects:
            b = self.bound_loc(obj)
            le = b[BOUND_LEFT]
            ri = b[BOUND_RIGHT]
            ba = b[BOUND_BACK]
            fr = b[BOUND_FRONT]
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]        
            ce = b[BOUND_CENTER]
            center += ce
            sample += 1
            top = max(top, to)
            bottom = min(bottom, bo)
        center /= sample
        """
            Detect body
        """
        min_length = sys.float_info.max
        body = None
        for obj in objects:
            b = self.bound_loc(obj)
            ce = b[BOUND_CENTER]
            length = (ce - center).length_squared
            if length < min_length:
                min_length = length
                body = obj
        if body is None:
            return
        """
            Detect others
        """
        heads = []
        right_arms = []    
        left_arms = []
        right_legs = []    
        left_legs = []
        hips_height = self.lerp(bottom, top, 0.333)
        for obj in objects:
            body_bound = self.bound_loc(body)
            body_center = body_bound[BOUND_CENTER]
            mat = Matrix(body.matrix_world)
            body_left = body_bound[BOUND_LEFT]
            body_right = body_bound[BOUND_RIGHT]
            body_radius = (body_left - body_right) * 0.5
            if body == obj:
                continue
            bound = self.bound_loc(obj)
            center = bound[BOUND_CENTER]
            radius = abs(center.x - body_center.x)
            if center.z < hips_height:
                if center.x < body_center.x:                
                    right_legs.append(obj)
                else:
                    left_legs.append(obj)
            elif radius < body_radius:
                heads.append(obj)
            else:
                if center.x < body_center.x:
                    right_arms.append(obj)
                else:
                    left_arms.append(obj)
        """
            Create armature
        """
        bpy.ops.object.armature_add(location=(0.0, 0.0, 0.0), enter_editmode=True)
        arma = context.active_object
        context.object.show_x_ray = True
        """
            Body bone
        """
        bone = arma.data.edit_bones[0]
        bone.name = "hips"
        hips, chest = self.create_bone(context, arma, bone, body, None, arma.data.edit_bones, bone_type=BONE_TYPE_BODY)
        """
            Leg bones
        """
        #for leg in right_legs:
        #    bone = arma.data.edit_bones.new("leg.R")
        #    CreateArmature.create_bone(context, arma, bone, leg, hips, arma.data.edit_bones, bone_type=CreateArmature.BONE_TYPE_LEG_RIGHT)
        #for leg in left_legs:        
        #    bone = arma.data.edit_bones.new("leg.L")
        #    CreateArmature.create_bone(context, arma, bone, leg, hips, arma.data.edit_bones, bone_type=CreateArmature.BONE_TYPE_LEG_LEFT)
        self.create_grouped_bone(context, right_legs, arma, hips, BONE_TYPE_LEG_RIGHT)
        self.create_grouped_bone(context, left_legs, arma, hips, BONE_TYPE_LEG_LEFT)

        head_groups = self.make_group(heads)
        for group in head_groups:
            primary_head = self.primary_obj(group)
            bone = arma.data.edit_bones.new("head")
            primary_bone = self.create_head(bone, primary_head, chest)
            for obj in group:
                if obj == primary_head:
                    continue
                bone = arma.data.edit_bones.new("head")
                self.create_bone(context, arma, bone, obj, primary_bone, arma.data.edit_bones, bone_type=BONE_TYPE_ANY)
        #for arm in right_arms:
        #    bone = arma.data.edit_bones.new("arm.R")
        #    CreateArmature.create_bone(context, arma, bone, arm, chest, arma.data.edit_bones, bone_type=CreateArmature.BONE_TYPE_ARM_RIGHT)
        #groups = CreateArmature.make_group(left_arms)
        #for group in groups:
        #    primary = CreateArmature.primary_obj(group)
        #    bone = arma.data.edit_bones.new("arm.L")
        #    k, primary_bone = CreateArmature.create_bone(context, arma, bone, primary, chest, arma.data.edit_bones, bone_type=CreateArmature.BONE_TYPE_ARM_LEFT)
        #    for obj in group:
        #        if obj == primary:
        #            continue
        #        bone = arma.data.edit_bones.new("arm.L")
        #        CreateArmature.create_bone(context, arma, bone, obj, primary_bone, arma.data.edit_bones, bone_type=CreateArmature.BONE_TYPE_ARM_LEFT)
        self.create_grouped_bone(context, right_arms, arma, chest, BONE_TYPE_ARM_RIGHT)
        self.create_grouped_bone(context, left_arms, arma, chest, BONE_TYPE_ARM_LEFT)
        if context.window_manager.love2d3d.armature_finger:
            """
                Rename fingers
            """
            left_fingers = []
            right_fingers = []
            for bone in arma.data.edit_bones:
                if bone.name.startswith("finger") and not bone.use_connect:
                    if bone.name.endswith(".L"):
                        left_fingers.append(bone)
                    else:
                        right_fingers.append(bone)
            lefts = sorted(left_fingers, key=lambda bone: bone.head.y)
            rights = sorted(right_fingers, key=lambda bone: bone.head.y)
            for k, bone4 in enumerate(lefts):
                bone4.name = "finger" + str(k) +".04" + ".L"
                bone3 = bone4.children_recursive[0]
                bone3.name =  "finger" + str(k) +".03" + ".L"
                bone2 = bone3.children_recursive[0]
                bone2.name =  "finger" + str(k) +".02" + ".L"
                bone1 = bone2.children_recursive[0]
                bone1.name =  "finger" + str(k) +".01" + ".L"
            for k, bone4 in enumerate(rights):
                bone4.name = "finger" + str(k) +".04" + ".R"
                bone3 = bone4.children_recursive[0]
                bone3.name =  "finger" + str(k) +".03" + ".R"
                bone2 = bone3.children_recursive[0]
                bone2.name =  "finger" + str(k) +".02" + ".R"
                bone1 = bone2.children_recursive[0]
                bone1.name =  "finger" + str(k) +".01" + ".R"


        for bone in arma.data.edit_bones:
            bone.select = True        
        bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')    
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        for obj in objects:
            obj.select = True
        context.scene.objects.active = arma
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        #print(datetime.datetime.today() - debug_time)
        return {'FINISHED'}

    def create_grouped_bone(self, context, objects, armature, parent, bone_type):
        groups = self.make_group(objects)
        for group in groups:
            primary = self.primary_obj(group)
            bone = armature.data.edit_bones.new("bone")
            k, primary_bone = self.create_bone(context, armature, bone, primary, parent, armature.data.edit_bones, bone_type=bone_type)
            for obj in group:
                if obj == primary:
                    continue
                bone = armature.data.edit_bones.new("bone")
                self.create_bone(context, armature, bone, obj, primary_bone, armature.data.edit_bones, bone_type=BONE_TYPE_ANY)
                
    def lerp(self, start, end , ratio):
        return start * (1 - ratio) + end * ratio

    def invlerp(self, value, start, end, r):
        ratio = (value - start) / (end - start)
        i = int(ratio * r + 0.5)
        #i = min(max(0, i), r - 1)
        i = min(max(0, i), r)
        return i

    def debug_point(self, context, location, type='PLAIN_AXES'):
        o = context.blend_data.objects.new("P", None)
        o.location = location
        o.scale = (0.01, 0.01, 0.01)
        context.scene.objects.link(o)
        o.empty_draw_type = type

    def create_bone(self, context, armature, bone, obj, chest, bones, bone_type=BONE_TYPE_BODY, fingers=None):
        #print(bone_type)
        finger = fingers is not None
        if finger:
            bones.remove(bone)
        mesh = obj.to_mesh(context.scene, True, 'PREVIEW')
        polygons = fingers if finger else mesh.polygons
        mat = Matrix(obj.matrix_world)
        if finger:
            polygons = fingers
            xs = [(mat * Vector(polygon.center)).x for polygon in polygons]
            ys = [(mat * Vector(polygon.center)).y for polygon in polygons]
            zs = [(mat * Vector(polygon.center)).z for polygon in polygons]

            #b = self.bound_loc(obj)
            #le = b[BOUND_LEFT]
            #ri = b[BOUND_RIGHT]
            #ba = b[BOUND_BACK]
            #fr = b[BOUND_FRONT]
            #to = b[BOUND_TOP]
            #bo = b[BOUND_BOTTOM]
            le = max(xs)
            ri = min(xs)
            ba = max(ys)
            fr = min(ys)
            to = max(zs)
            bo = min(zs)
        else:
            polygons = mesh.polygons
            b = self.bound_loc(obj)
            le = b[BOUND_LEFT]
            ri = b[BOUND_RIGHT]
            ba = b[BOUND_BACK]
            fr = b[BOUND_FRONT]
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]
        ce = Vector((self.lerp(ri, le, 0.5), self.lerp(fr, ba, 0.5), self.lerp(bo, to, 0.5)))
        if bone_type == BONE_TYPE_BODY:
            #ce = b[BOUND_CENTER]
            body_top = Vector((ce.x, ce.y, to))
            body_bottom = Vector((ce.x, ce.y, bo))
        len_x = le - ri
        len_y = ba - fr
        len_z = to - bo
        armature_resolution = context.window_manager.love2d3d.armature_resolution
        if finger:
            #digit = np.floor(np.log2(armature_resolution)) + 1
            #lattice = min(len_x, len_y, len_z) / np.power(2.0, digit) # latiice unit
            #lattice = min(len_x, len_y, len_z) / .0 # latiice unit
            lattice = min(len_x, len_y, len_z) / context.window_manager.love2d3d.armature_finger_resolution # latiice unit
        else:
            lattice = min(len_x, len_y, len_z) / armature_resolution # latiice unit
        #if lattice == 0.0:
        #print(obj.name)
        #print("{}, {}, {}".format(len_x, len_y, len_z))
        if lattice == 0.0:
            return None, None
        rx = int(len_x / lattice) # x loop count
        ry = int(len_y / lattice) # y loop count
        rz = int(len_z / lattice) # z loop count    
        rx = max(1, rx)
        ry = max(1, ry)
        rz = max(1, rz)    
        mx = 1.0 / float(rx)
        my = 1.0 / float(ry)
        mz = 1.0 / float(rz)
        start = (0, 0, 0)
        min_dist = sys.float_info.max
        if bone_type == BONE_TYPE_BODY:
            origin = body_bottom
        else:
            origin = Vector(chest.tail)    
        centers = []
        """
            Volume separation process to reduce polygons' calculation.
        """
        #x0_y0_z0_polygons = []    
        #x1_y0_z0_polygons = []
        #x0_y1_z0_polygons = []
        #x0_y0_z1_polygons = []    
        #x0_y1_z1_polygons = []
        #x1_y0_z1_polygons = []
        #x1_y1_z0_polygons = []
        #x1_y1_z1_polygons = []
        #half_x = self.lerp(ri, le, 0.5)
        #half_y = self.lerp(fr, ba, 0.5)
        #half_z = self.lerp(bo, to, 0.5)
        #x0_polygons = []
        #x1_polygons = []
        #y0_polygons = []
        #y1_polygons = []
        #z0_polygons = []
        #z1_polygons = []
                
        """
            Angle-based separation
        """
        #loop = 10
        #cakes = [[[[] for u in range(loop * 2 + 1)] for v in range(loop * 2 + 1)] for w in range(loop * 2 + 1)]
        ##kds = [[[None for u in range(loop + 1)] for v in range(loop + 1)] for w in range(loop + 1)]
        #for polygon in polygons: # Nearest polygon
        #    center =  mat * Vector(polygon.center)
        #    n = (center - ce).normalized()
        #    u = int(np.round(n.x * loop)) + loop
        #    v = int(np.round(n.y * loop)) + loop
        #    w = int(np.round(n.z * loop)) + loop
        #    #print("{},{},{}".format(u, v, w))
        #    cakes[w][v][u].append(polygon)
        ##        for cake in cakes:
        ##            kd = KDTree(len(cake))
        ##            for i, polygon in enumerate(cake):
        ##                kd.insert(mat * Vector(polygon.center), i)
        ##            kd.balance()
        ##            kds = 

        loop = 2
        half_x = self.lerp(ri, le, 0.5)
        half_y = self.lerp(fr, ba, 0.5)
        half_z = self.lerp(bo, to, 0.5)
        #half_x = (le - ri) / float(loop)
        #half_y = (ba - fr) / float(loop)
        #half_z = (to - bo) / float(loop)
        #cake_xs = [[] for x in range(loop)]
        #cake_ys = [[] for y in range(loop)]
        #cake_zs = [[] for z in range(loop)]
        cakes = [[[[] for x in range(loop)] for y in range(loop)]for z in range(loop)]
        for polygon in polygons: # Nearest polygon
            center =  mat * Vector(polygon.center)
            x = 0 if center.x <= half_x else 1
            y = 0 if center.y <= half_y else 1
            z = 0 if center.z <= half_z else 1
            #x = int((center.x - ri) / half_x)
            #y = int((center.y - fr) / half_y)
            #z = int((center.z - bo) / half_z)
            cakes[z][y][x].append(polygon)
            #cake_xs[x].append(polygon)
            #cake_ys[y].append(polygon)
            #cake_zs[z].append(polygon)

        kds = [[[None for x in range(loop)] for y in range(loop)] for z in range(loop)]
        for x in range(loop):
            for y in range(loop):
                for z in range(loop):
                    cake = cakes[z][y][x]
                    kd = KDTree(len(cake))
                    for i, polygon in enumerate(cake):
                        kd.insert(mat * Vector(polygon.center), i)
                    kd.balance()
                    kds[z][y][x] = kd

        #bvhs = [[[None for x in range(loop)] for y in range(loop)] for z in range(loop)]
        #for x in range(loop):
        #    for y in range(loop):
        #        for z in range(loop):
        #            cake = cakes[z][y][x]
        #            vs = []
        #            ps = []
        #            offset = 0
        #            for polygon in cake:
        #                vs.extend([(mat * Vector(mesh.vertices[vertice].co)).xyz for vertice in polygon.vertices])
        #                ps.append(range(offset, polygon.loop_total + offset))
        #                offset += polygon.loop_total
        #            bvh = BVHTree.FromPolygons(vs, ps)
        #            #for i, polygon in enumerate(cake):
        #            #    kd.insert(mat * Vector(polygon.center), i)
        #            #kd.balance()
        #            #kds[z][y][x] = kd
        #            bvhs[z][y][x] = bvh

        #for polygon in polygons: # Nearest polygon
        #    center =  mat * Vector(polygon.center)
        #    if center.x <= half_x:
        #        if center.y <= half_y:
        #            if center.z <= half_z:
        #                x0_y0_z0_polygons.append(polygon)
        #                z0_polygons.append(polygon)
        #            else:
        #                x0_y0_z1_polygons.append(polygon)
        #                z1_polygons.append(polygon)
        #            y0_polygons.append(polygon)
        #        else:
        #            if center.z <= half_z:
        #                x0_y1_z0_polygons.append(polygon)                    
        #                z0_polygons.append(polygon)
        #            else:
        #                x0_y1_z1_polygons.append(polygon)
        #                z1_polygons.append(polygon)

        #            y1_polygons.append(polygon)
        #        x0_polygons.append(polygon)
        #    else:
        #        if center.y <= half_y:
        #            if center.z <= half_z:
        #                x1_y0_z0_polygons.append(polygon)                    
        #                z0_polygons.append(polygon)
        #            else:                
        #                x1_y0_z1_polygons.append(polygon)
        #                z1_polygons.append(polygon)
        #            y0_polygons.append(polygon)
        #        else:
        #            if center.z <= half_z:
        #                x1_y1_z0_polygons.append(polygon)
        #                z0_polygons.append(polygon)
        #            else:
        #                x1_y1_z1_polygons.append(polygon)
        #                z1_polygons.append(polygon)
        #            y1_polygons.append(polygon)
        #        x1_polygons.append(polygon)
        #s = len(x0_y0_z0_polygons)
        #x0_y0_z0_kd = KDTree(s)
        #for i, polygon in enumerate(x0_y0_z0_polygons):
        #    x0_y0_z0_kd.insert(mat * Vector(polygon.center), i)
        #x0_y0_z0_kd.balance()

        #s = len(x0_y0_z1_polygons)
        #x0_y0_z1_kd = KDTree(s)
        #for i, polygon in enumerate(x0_y0_z1_polygons):
        #    x0_y0_z1_kd.insert(mat * Vector(polygon.center), i)
        #x0_y0_z1_kd.balance()

        #s = len(x0_y1_z0_polygons)
        #x0_y1_z0_kd = KDTree(s)
        #for i, polygon in enumerate(x0_y1_z0_polygons):
        #    x0_y1_z0_kd.insert(mat * Vector(polygon.center), i)
        #x0_y1_z0_kd.balance()

        #s = len(x0_y1_z1_polygons)
        #x0_y1_z1_kd = KDTree(s)
        #for i, polygon in enumerate(x0_y1_z1_polygons):
        #    x0_y1_z1_kd.insert(mat * Vector(polygon.center), i)
        #x0_y1_z1_kd.balance()

        #s = len(x1_y0_z0_polygons)
        #x1_y0_z0_kd = KDTree(s)
        #for i, polygon in enumerate(x1_y0_z0_polygons):
        #    x1_y0_z0_kd.insert(mat * Vector(polygon.center), i)
        #x1_y0_z0_kd.balance()

        #s = len(x1_y0_z1_polygons)
        #x1_y0_z1_kd = KDTree(s)
        #for i, polygon in enumerate(x1_y0_z1_polygons):
        #    x1_y0_z1_kd.insert(mat * Vector(polygon.center), i)
        #x1_y0_z1_kd.balance()

        #s = len(x1_y1_z0_polygons)
        #x1_y1_z0_kd = KDTree(s)
        #for i, polygon in enumerate(x1_y1_z0_polygons):
        #    x1_y1_z0_kd.insert(mat * Vector(polygon.center), i)
        #x1_y1_z0_kd.balance()

        #s = len(x1_y1_z1_polygons)
        #x1_y1_z1_kd = KDTree(s)
        #for i, polygon in enumerate(x1_y1_z1_polygons):
        #    x1_y1_z1_kd.insert(mat * Vector(polygon.center), i)
        #x1_y1_z1_kd.balance()

        #co_find = (0.0, 0.0, 0.0)
        #co, index, dist = x0_y0_z0_kd.find(co_find)



        #co_find = (0.0, 0.0, 0.0)
        #co, index, dist = x0_y0_z0_kd.find(co_find)
        
        #dtheta = 0.1
        #loop = int(360.0 / dtheta) + 1
        #cakes = [[] for l in range(loop)]
        #for polygon in polygons:
        #    center =  mat * Vector(polygon.center)
        #    theta = np.arctan2(center.y - ce.y, center.x - ce.x)
        #    index = int((np.degrees(theta) + 180) / dtheta)
        #    cakes[index].append(polygon)
        #    print(index)
#        kd = KDTree(len(polygons))
#        for i, polygon in enumerate(polygons):
#            kd.insert(mat * Vector(polygon.center), i)
#        kd.balance()

        """
            Deciding process of inside points.
        """
        #xs = []
        #for x in range(rx + 1):
        #    #hit = False
        #    pick = self.lerp(ri, le, x * mx)
        #    u = 0 if pick <= half_x else 1
        #    for polygon in cake_xs[u]:
        #        center =  mat * Vector(polygon.center)
        #        current = Vector((pick, center.y, center.z))
        #        normal = mat.to_quaternion() * Vector(polygon.normal)
        #        radius = np.sqrt(polygon.area) * 0.5
        #        vec = current - center
        #        coeff = vec.dot(normal) #Projection to center along Normal
        #        vec -= coeff * normal
        #        length = vec.length_squared
        #        #length = vec.length
        #        #hit = hit or (length < radius and coeff < 0)
        #        if length < radius and coeff < 0:
        #            xs.append(x)
        #            break
        #    #if hit:
        #    #    xs.append(x)
        #ys = []
        #for y in range(ry + 1):
        #    #hit = False
        #    #polygons = []
        #    pick = self.lerp(fr, ba, y * my)
        #    v = 0 if pick <= half_y else 1
        #    for polygon in cake_ys[v]:
        #        center =  mat * Vector(polygon.center)
        #        current = Vector((center.x, pick, center.z))
        #        normal = mat.to_quaternion() * Vector(polygon.normal)
        #        radius = np.sqrt(polygon.area) * 0.5
        #        vec = current - center
        #        coeff = vec.dot(normal) #Projection to center along Normal
        #        vec -= coeff * normal
        #        length = vec.length_squared
        #        #length = vec.length
        #        #hit = hit or (length < radius and coeff < 0)
        #        if length < radius and coeff < 0:
        #            ys.append(y)
        #            break
        #    #if hit:
        #    #    ys.append(y)
        #zs = []
        #for z in range(rz + 1):
        #    #hit = False
        #    #polygons = []
        #    pick = self.lerp(bo, to, z * mz)
        #    w = 0 if pick <= half_z else 1
        #    for polygon in cake_zs[w]:
        #        center =  mat * Vector(polygon.center)
        #        current = Vector((center.x, center.y, pick))
        #        normal = mat.to_quaternion() * Vector(polygon.normal)
        #        radius = np.sqrt(polygon.area) * 0.5
        #        vec = current - center
        #        coeff = vec.dot(normal) #Projection to center along Normal
        #        vec -= coeff * normal
        #        length = vec.length_squared
        #        #length = vec.length
        #        #hit = hit or (length < radius and coeff < 0)
        #        if length < radius and coeff < 0:
        #            zs.append(z)
        #            break
        #    #if hit:
        #    #    zs.append(z)

        for x in range(rx + 1):
            for y in range(ry + 1):
                for z in range(rz + 1):
                    current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                    current_length = sys.float_info.max
                    current_height = sys.float_info.max
                    #polygons = []
                    #kd = None
                    #if current.x <= half_x:
                    #    if current.y <= half_y:
                    #        if current.z <= half_z:
                    #            polygons = x0_y0_z0_polygons
                    #            kd = x0_y0_z0_kd
                    #        else:
                    #            polygons = x0_y0_z1_polygons
                    #            kd = x0_y0_z1_kd
                    #    else:
                    #        if current.z <= half_z:
                    #            polygons = x0_y1_z0_polygons
                    #            kd = x0_y1_z0_kd
                    #        else:
                    #            polygons = x0_y1_z1_polygons
                    #            kd = x0_y1_z1_kd
                    #else:
                    #    if current.y <= half_y:
                    #        if current.z <= half_z:
                    #            polygons = x1_y0_z0_polygons
                    #            kd = x1_y0_z0_kd
                    #        else:
                    #            polygons = x1_y0_z1_polygons
                    #            kd = x1_y0_z1_kd
                    #    else:
                    #        if current.z <= half_z:
                    #            polygons = x1_y1_z0_polygons
                    #            kd = x1_y1_z0_kd
                    #        else:
                    #            polygons = x1_y1_z1_polygons
                    #            kd = x1_y1_z1_kd

                    #n = (current - ce).normalized()
                    #u = int(np.round(n.x * loop)) + loop
                    #v = int(np.round(n.y * loop)) + loop
                    #w = int(np.round(n.z * loop)) + loop
                    #polygons = cakes[w][v][u]
                    #print("{},{},{}".format(u, v, w))

                    min_polygon = None

                    #n = int((current.x - ri) / half_x)
                    #m = int((current.y - fr) / half_y)
                    #l = int((current.z - bo) / half_z)
                    n = 0 if current.x <= half_x else 1
                    m = 0 if current.y <= half_y else 1
                    l = 0 if current.z <= half_z else 1
                    #cakes[z][y][x].append(polygon)
                    co_find = (current.x, current.y, current.z)
                    #if kds[l][m][n] is None:
                    #    continue
                    #print(kds[l][m][n])
                    co, index, dist = kds[l][m][n].find(co_find)
                    #co, index, dist = kd.find(co_find)

#                    location, normal, index, distance = bvhs[l][m][n].find_nearest(co_find)

                    #print("{},{},{},{}".format(l, m, n, index))
                    #if index is None:                    
                    #    continue
                    if index is None:                    
                        continue
                    min_polygon = cakes[l][m][n][index]
                    #min_polygon = polygons[index]

                    #co_find = (current.x, current.y, current.z)
                    #co, index, dist = kd.find(co_find)
                    #min_polygon = polygons[index]

                    #min_length = sys.float_info.max
                    #for polygon in polygons: # Nearest polygon
                    #    center =  mat * Vector(polygon.center)
                    #    length = (current - center).length_squared
                    #    if length < min_length:
                    #        min_polygon = polygon
                    #        min_length = length

                    if min_polygon is None:                    
                        continue
                    normal = mat.to_quaternion() * Vector(min_polygon.normal)                
                    center = mat * (min_polygon.center)
                    """
                        Approximation of polygon's region.
                    """
                    min_length = np.sqrt(min_polygon.area) * 0.5
                    vec = current - center
                    coeff = vec.dot(normal) #Projection to center along Normal
                    vec -= coeff * normal
                    length = vec.length_squared # This is mistake. But this is good result.
                    #length = vec.length
                    close = coeff
                    if length < min_length and close < 0:
                        #loc = coeff * normal + center
                        #lx, ly, lz = loc.xyz
                        #u = self.invlerp(lx, ri, le, rx)
                        #v = self.invlerp(ly, fr, ba, ry)
                        #w = self.invlerp(lz, bo, to, rz)
                        #centers.append((u, v, w)) # Volume thinning
                        if finger:
                            loc = coeff * normal + center
                            lx, ly, lz = loc.xyz
                            u = self.invlerp(lx, ri, le, rx)
                            v = self.invlerp(ly, fr, ba, ry)
                            w = self.invlerp(lz, bo, to, rz)
                            centers.append((u, v, w)) # Volume thinning
                        else:
                            centers.append((x, y, z))
                    #centers.append((x, y, z))
                    #print("{},{},{}".format(x, y, z))
                        #self.debug_point(context, loc)
                        #self.debug_point(context, current, type="CUBE")
                        #centers.append((x, y, z))

                    
        """        
            Getting process of the nearest point from origin.
        """
        center_kd = KDTree(len(centers))
        for i, c in enumerate(centers):
            x, y, z = c
            current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
            center_kd.insert(current.xyz, i)
        center_kd.balance()

        #min_dist = sys.float_info.max
        #start = (0, 0, 0)
        #for c in centers:
        #    x, y, z = c         
        #    current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
        #    dist = (current - origin).length_squared
        #    if dist < min_dist:
        #        min_dist = dist
        #        start = (x, y, z)
        co, index, dist = center_kd.find(origin.xyz)
        #cx, cy, cz = centers[index]
        start = centers[index]
        x, y, z = start
        current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
        if not finger:
            bone.head = current
        """
            Getting process of the farthest point.
        """
        end = start
        max_length = 0.0
        max_coeff = 0.0
        for c in centers:
            x, y, z = c
            current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
            u, v , w = start
            s =  Vector((self.lerp(ri, le, u * mx), self.lerp(fr, ba, v * my), self.lerp(bo, to, w * mz)))
            if finger:
                vec = chest.tail - chest.head
                #length = (current - s).length_squared
                coeff = (current - s).dot(vec)
                if max_coeff < coeff:
                    max_coeff = coeff
                    end = (x, y, z)
            else:
                length = (current - s).length_squared
                if max_length < length:
                    max_length = length
                    end = (x, y, z)
        sx, sy, sz = start

        def limit_hips(index):
            stem = (body_top - body_bottom)
            cx, cy, cz = centers[index]
            current = Vector((self.lerp(ri, le, cx * mx), self.lerp(fr, ba, cy * my), self.lerp(bo, to, cz * mz)))
            branch = current - body_bottom
            if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                return False
            return stem.angle(branch) < self.hips_limit_angle

        #neck_limit = BRANCH_LIMIT_HIPS
        if bone_type == BONE_TYPE_BODY:
            min_length = sys.float_info.max
            min_loc = body_bottom
            """
                Getting process of hips point.
            """        
            #for c in centers:
            #    x, y, z = c
            #    current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
            #    length = (current - body_bottom).length_squared
            #    stem = (body_top - body_bottom)
            #    branch0 = (current - body_bottom)
            #    length = branch0.length_squared            
            #    if stem.length_squared == 0.0 or branch0.length_squared == 0.0:
            #        continue            
            #    if length < min_length and stem.angle(branch0) < self.hips_limit_angle:
            #        min_length = length
            #        min_loc = current            
            co, index, dist = center_kd.find(origin.xyz, filter=limit_hips)
            #start = centers[index]
            #start_loc = Vector((body_bottom.x, body_bottom.y, min_loc.z))
            #cx, cy, cz =  centers[index]
            #print(index)
            #print(co)
            if co is None:
                start_loc = body_bottom
            else:
                start_loc = Vector((body_bottom.x, body_bottom.y, Vector(co).z))
            #start_loc = body_bottom
        else:
            start_loc = Vector((self.lerp(ri, le, sx * mx), self.lerp(fr, ba, sy * my), self.lerp(bo, to, sz * mz)))
        if not finger:
            bone.head = start_loc
        ex, ey, ez = end
        if bone_type == BONE_TYPE_BODY:        
            end_loc = body_top
        else:
            end_loc = Vector((self.lerp(ri, le, ex * mx), self.lerp(fr, ba, ey * my), self.lerp(bo, to, ez * mz)))
        """
            Getting process of center and neck point.
        """                
        #m = (start_loc + end_loc) * 0.5
        #min_length = sys.float_info.max    
        #center_loc = start_loc
        #
        #n = start_loc.lerp(end_loc, 0.75)
        #neck_length = sys.float_info.max
        #neck_loc = start_loc
        #for center in centers:
        #    cx, cy, cz = center
        #    current = Vector((self.lerp(ri, le, cx * mx), self.lerp(fr, ba, cy * my), self.lerp(bo, to, cz * mz)))
        #    m_length = (current - m).length_squared
        #    n_length = (current - n).length_squared
        #    if m_length < min_length:
        #        center_loc = current
        #        min_length = m_length
        #    if n_length < neck_length:
        #        neck_loc = current
        #        neck_length = n_length
        #
        co, index, dist = center_kd.find(start_loc.lerp(end_loc, 0.5).xyz)
        #cx, cy, cz = centers[index]
        center_loc = co
        co, index, dist = center_kd.find(start_loc.lerp(end_loc, 0.75).xyz)
        #cx, cy, cz = centers[index]
        neck_loc = co
        if bone_type == BONE_TYPE_BODY:        
            hips_loc = start_loc.lerp(end_loc, 0.25)
            bone.tail = hips_loc
            center_loc = start_loc.lerp(end_loc, 0.5)
            neck_loc = start_loc.lerp(end_loc, 0.75)
        elif finger:
            pass
        else:
            bone.tail = center_loc        
            bone.parent = chest        
        parent = bone
        start_bone = bone
    
        """
            Create process of primary bones.
        """                    
        if bone_type == BONE_TYPE_BODY:        
            bone = bones.new("waist")
            bone.head = hips_loc
            bone.tail = center_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            center_bone = bone
        
            bone = bones.new("chest")
            bone.head = center_loc
            bone.tail = neck_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            chest_bone = bone
        
            bone = bones.new("neck")
            bone.head = neck_loc
            bone.tail = end_loc
            bone.parent = parent
            bone.use_connect = True    
            end_bone = bone

        elif bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
            name = "finger"
            end_bone = None
            start_bone = None
            #pass
        else:
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "shoulder.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "shoulder.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "pelvis.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "pelvis.R"
            else:
                name = "bone"
            bone.name = name
            vec = (origin - start_loc)
            basis = (start_loc - center_loc).normalized()
            coeff = vec.dot(basis)
            bone.head = 0.5 * coeff * basis + start_loc
            bone.tail = start_loc

            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "upper_arm.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "upper_arm.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "thigh.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "thigh.R"
            else:
                name = "bone"
            bone = bones.new(name)
            bone.head = start_loc
            bone.tail = center_loc        
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            center_bone = bone

        
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "forearm.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "forearm.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "shin.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "shin.R"
            else:
                name = "bone"    
            bone = bones.new(name)
            bone.head = center_loc
            bone.tail = neck_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            center_bone = bone
        
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "hand.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "hand.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "foot.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "foot.R"
            else:
                name = "bone"
            bone = bones.new(name)
            bone.head = neck_loc
            bone.tail = end_loc
            bone.parent = parent
            bone.use_connect = True    
            end_bone = bone

        """
            Caluculate process of volume.
        """    
        volume_xs = [0 for x in range(rx + 1)]
        volume_ys = [0 for y in range(ry + 1)]
        volume_zs = [0 for z in range(rz + 1)]
        soft  = 1
        unit_x = my * mz
        unit_y = mz * mx
        unit_z = mx * my
        for center in centers:
            x, y, z = center
            volume_xs[x] += unit_x
            volume_ys[y] += unit_y
            volume_zs[z] += unit_z            
        hit_xs = []
        hit_ys = []
        hit_zs = []
        threshopld = 1.0
        ratio = self.finger_branch_boost if finger else self.branch_boost
        soft_x = unit_x * 0.0001 
        soft_y = unit_y * 0.0001
        soft_z = unit_z * 0.0001
        """
            Differential process of volume in log scale.
            It tell us like "A's scale is B's scale of x1, x10, x100...".
        """
        for x in range(1, rx + 1 - 1):
            vm = volume_xs[x - 1]
            v0 = volume_xs[x]
            v1 =  volume_xs[x + 1]
            sv = np.log2(v0 + soft_x) * ratio
            ev = np.log2(v1 + soft_x) * ratio
            mv = np.log2(vm + soft_x) * ratio        
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_xs.append(x)
        for y in range(1, ry + 1 - 1):
            vm = volume_ys[y - 1]        
            v0 = volume_ys[y]
            v1 = volume_ys[y + 1]
            sv = np.log2(v0 + soft_y) * ratio
            ev = np.log2(v1 + soft_y) * ratio
            mv = np.log2(vm + soft_y) * ratio        
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_ys.append(y)
        for z in range(1, rz + 1 - 1):
            vm = volume_zs[z - 1]        
            v0 = volume_zs[z]
            v1 = volume_zs[z + 1]
            sv = np.log2(v0 + soft_z) * ratio
            ev = np.log2(v1 + soft_z) * ratio
            mv = np.log2(vm + soft_z) * ratio        
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_zs.append(z)
        """
            Diciding process of branch point like fingers.
        """
        hits = []
        average = Vector((0, 0, 0))
        dispersion = 0.0    
        sum = 0
        center_limit = self.center_limit_angle
        for x in hit_xs:
            for y in hit_ys:
                for z in hit_zs:
                    for center in centers:
                        u, v, w = center
                        current = Vector((self.lerp(ri, le, u * mx), self.lerp(fr, ba, v * my), self.lerp(bo, to, w * mz)))
                        free = True
                        """
                            Avoiding process of body's neck.
                            It is because bad points for shoulders.
                        """
                        if bone_type == BONE_TYPE_BODY:                        
                            proj0 = Vector((current.x, body_bottom.y, current.z))
                            proj1 = Vector((current.x, body_top.y, current.z))                        
                            branch0 = (proj0 - body_bottom)
                            branch1 = (proj1 - body_top)
                            stem = (body_top - body_bottom)
                            if branch0.length_squared == 0.0 or branch1.length_squared == 0.0:
                                continue
                            angle0 = stem.angle(branch0)
                            angle1 = stem.angle(branch1)                        
                            free = center_limit < angle0 and center_limit < angle1
                        if x == u and y == v and z == w and free:
                            hits.append((u, v, w))
                            average += current
                            dispersion += current.length_squared
                            sum += 1
#                            if finger:
#                                self.debug_point(context, current, type="CUBE")
#                            if finger:
#                                self.debug_point(context, current)


        """
            Gathering process of branch points.
        """    
        if sum == 0:
            return start_bone, end_bone
        average /= sum
        dispersion /= sum
        dispersion -= average.length_squared

        #dispersion *= (self.finger_gather_ratio) if finger else (self.gather_ratio* 0.01)
        gather_ratio = (dispersion * self.finger_gather_ratio * 0.01) if finger else (dispersion * self.gather_ratio * 0.01)
        bound = le, ri, ba, fr, to, bo
        m = mx, my, mz
        gathers = self.gather_point(hits, bound, m, gather_ratio)
        possible_joints = []
        joints = []
        """
            Averaging process of gatherd branch points.
        """
        for gather in gathers:
            average = Vector((0, 0, 0))
            sum = 0
            for point in gather:
                px, py, pz = hits[point]
                p_loc = Vector((self.lerp(ri, le, px * mx), self.lerp(fr, ba, py * my), self.lerp(bo, to, pz * mz)))
                average += p_loc
                sum += 1
            if sum == 0:
                continue
            average /= sum
            joints.append(average)
#            if finger:
#                self.debug_point(context, average, type="CUBE")
#            if finger:
#                self.debug_point(context, average)

        """
            Calculating and creating process of bones.
        """
        def create_joint(centers, hinges, bounds, rs, ms, dispersion, name, parent, bone_type, end=Vector((0, 0, 0))):
            le, ri, ba, fr, to, bo = bounds
            rx, ry, rz = rs
            ms = (mx, my, mz)
            tips = [(self.invlerp(hinge[1].x, ri, le, rx), self.invlerp(hinge[1].y, fr, ba, ry), self.invlerp(hinge[1].z, bo, to, rz)) for hinge in hinges]
            if bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
                 gathers = self.gather_point(tips, bounds, ms, dispersion, parent=parent, end=end)
            else:
                 gathers = self.gather_point(tips, bounds, ms, dispersion)
            """
                Averaging process of gathered tips.
            """        
            averages = [Vector((0,0,0)) for g in gathers]
            for k, gather in enumerate(gathers):
                if bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
                    average = Vector((0, 0, 0))
                    #sum = 0
                    max_close = 0.0
                    max_point = Vector((0, 0, 0))
                    vec = end - parent.head
                    for point in gather:
                        px, py, pz = tips[point]
                        p_loc = Vector((self.lerp(ri, le, px * mx), self.lerp(fr, ba, py * my), self.lerp(bo, to, pz * mz)))
                        close = p_loc.dot(vec)
                        if max_close < close:
                            max_close = close
                            max_point = p_loc
                        #average += p_loc
                        #sum += 1
                    #average /= sum
                    averages[k] = max_point
    #                if finger:
                else:
                    average = Vector((0, 0, 0))
                    sum = 0
                    for point in gather:
                        px, py, pz = tips[point]
                        p_loc = Vector((self.lerp(ri, le, px * mx), self.lerp(fr, ba, py * my), self.lerp(bo, to, pz * mz)))
                        average += p_loc
                        sum += 1
                    average /= sum
                    averages[k] = average
#                self.debug_point(context, average, type="CUBE")
            """
                Grouping hinge by average points.
            """
            groups = [[] for a in averages]
            for hinge in hinges:
                joint, tip = hinge
                min_length = sys.float_info.max
                min_index = 0
                for k, average in enumerate(averages):
                    length = (average - tip).length_squared
                    if length < min_length:
                        min_length = length
                        min_index = k
                groups[min_index].append(joint)
            #bone = None
            end_bones = []
            for k, tip in enumerate(averages):
                max_length = -sys.float_info.max
                max_joint = tip
                group = groups[k]
                if len(group) == 0:
                    continue
                for joint in group:
                    length = (joint - tip).length_squared
                    if max_length < length:
                        max_length = length
                        max_joint = joint
                #e = max_joint.lerp(tip, 0.5)
                #n = max_joint.lerp(tip, 0.75)
                #t = max_joint.lerp(tip, 0.875)
                #min_e = Vector((0, 0, 0))
                #min_n = Vector((0, 0, 0))
                #min_t = Vector((0, 0, 0))
                #min_e_length = sys.float_info.max
                #min_n_length = sys.float_info.max
                #min_t_length = sys.float_info.max
                #for center in centers:
                #    x, y, z = center
                #    current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                #    e_length = (current - e).length_squared
                #    n_length = (current - n).length_squared
                #    t_length = (current - t).length_squared
                #    if e_length < min_e_length:
                #        min_e_length = e_length
                #        min_e = current
                #    if n_length < min_n_length:
                #        min_n_length = n_length
                #        min_n = current
                #    if t_length < min_t_length:
                #        min_t_length = t_length
                #        min_t = current
                """
                    Elbow
                """
                co, index, dist = center_kd.find(max_joint.lerp(tip, 0.5).xyz)
                #cx, cy, cz = centers[index]
                min_e = co
                """
                    Hand neck
                """
                co, index, dist = center_kd.find(max_joint.lerp(tip, 0.75).xyz)
                #cx, cy, cz = centers[index]
                min_n = co
                """
                    Finger tip
                """
                co, index, dist = center_kd.find(max_joint.lerp(tip, 0.875).xyz)
                #cx, cy, cz = centers[index]
                min_t = co
                """
                    Shoulder
                """
                vec = (max_joint - min_e).normalized()
                coeff = (parent.tail - max_joint).dot(vec)
                s = vec * coeff * 0.5 + max_joint
                #min_s = Vector((0, 0, 0))
                #min_s_length = sys.float_info.max
                #for center in centers:
                #    x, y, z = center
                #    current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                #    s_length = (current - s).length_squared            
                #    if s_length < min_s_length:
                #        min_s_length = s_length
                #        min_s = current
                co, index, dist = center_kd.find(s.xyz)
                #cx, cy, cz = centers[index]
                min_s = co

                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "shoulder.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "shoulder.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "pelvis.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "pelvis.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"

                if bone_type != BONE_TYPE_FINGER_LEFT and bone_type != BONE_TYPE_FINGER_RIGHT:
                    bone = bones.new(name)
                    bone.head = min_s
                    bone.tail = max_joint
                    bone.parent = parent
                    p = bone

                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "upper_arm.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "upper_arm.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "thigh.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "thigh.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
            
                bone = bones.new(name)
                bone.head = max_joint
                bone.tail = min_e
                if bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
                    bone.parent = parent
                else:
                    bone.parent = p
                    bone.use_connect = True
                p = bone
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "forearm.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "forearm.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "shin.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "shin.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
            
                bone = bones.new(name)
                bone.head = min_e
                bone.tail = min_n
                bone.parent = p
                bone.use_connect = True                
                p = bone
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "hand.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "hand.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "foot.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "foot.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
                bone = bones.new(name)
                bone.head = min_n
                if bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
                    bone.tail = min_t
                else:
                    bone.tail = tip
                bone.parent = p
                bone.use_connect = True
                p = bone
                if bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                    bone = bones.new(name)
                    bone.head = min_n
                    bone.tail = tip
                    bone.parent = p
                    bone.use_connect = True
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                    bone = bones.new(name)
                    bone.head = min_n
                    bone.tail = tip
                    bone.parent = p
                    bone.use_connect = True
                end_bones.append(bone)

            return end_bones
        """
            Getting process of tips like fingers's tip.
        """
        left_hands = []
        right_hands = []
        left_foots = []
        right_foots = []

        if bone_type == BONE_TYPE_BODY:
            right_arms = []
            left_arms = []
            right_legs = []
            left_legs = []
            for joint in joints:
                arm = hips_loc.z < joint.z
                left = 0 < joint.x - center_loc.x
                if arm:
                    stem = neck_loc - center_loc
                    branch = joint - center_loc
                else:
                    stem = start_loc - center_loc
                    branch = joint - center_loc
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                limit_angle = self.arm_limit_angle if arm else self.leg_limit_angle
                if limit_angle < angle:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    for center in centers:
                        x, y, z = center
                        current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                        if arm:
                            height = hips_loc.z < current.z
                        else:
                            height = current.z < hips_loc.z
                        close =  (current - joint).dot(branch_normal)
                        if height and max_close < close:
                            max_close = close
                            max_loc = current
                    if arm:
                        if left:
                            left_arms.append((joint, max_loc))                        
                        else:
                            right_arms.append((joint, max_loc))
                    else:
                        if left:
                            left_legs.append((joint, max_loc))                        
                        else:
                            right_legs.append((joint, max_loc))                        
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            left_hands = create_joint(centers, left_arms, bounds, rs, ms, gather_ratio, "Arm.L", chest_bone, BONE_TYPE_ARM_LEFT)
            right_hands = create_joint(centers, right_arms, bounds, rs, ms, gather_ratio, "Arm.R", chest_bone, BONE_TYPE_ARM_RIGHT)
            left_foots = create_joint(centers, left_legs, bounds, rs, ms, gather_ratio, "Leg.L", start_bone, BONE_TYPE_LEG_LEFT)
            right_foots = create_joint(centers, right_legs, bounds, rs, ms, gather_ratio, "Leg.R", start_bone, BONE_TYPE_LEG_RIGHT)
        elif bone_type == BONE_TYPE_FINGER_LEFT or bone_type == BONE_TYPE_FINGER_RIGHT:
            tips = []
            for joint in joints:
                #stem = chest.tail - chest.head
                stem = end_loc - chest.head
                #end_loc
                #branch = joint - chest.head.lerp(hand_center, 1.0)
                branch = joint - chest.head
                #branch = joint - hand_center
                #hand_center.dot()
#                branch1 = joint - chest.head.lerp(chest.tail, 0.5)
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                #limit_angle = BRANCH_LIMIT_FINGER
                limit_angle = self.finger_limit_angle
                close = stem.dot(branch)
#                close1 = branch1.dot(stem)
               
                if 0 < close:
#                if 0 < close and 0 < close1:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    #hit = False
                    min_angle = sys.float_info.max
                    for center in centers:
                        x, y, z = center
                        current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                        vec0 = current - joint
                        #vec1 = current - end_loc
                        #vec1 = current - chest.head.lerp(chest.tail, 0.5)
                        close =  vec0.dot(stem)
                        if vec0.length_squared == 0.0:
                            continue
                        angle = vec0.angle(stem)
                        # and 0 < vec1.dot(stem)
                        if max_close < close and angle < limit_angle:
                        #if angle < min_angle:
                            max_close = close
                            max_loc = current
                            min_angle = angle
                    tips.append((joint, max_loc))                        
#                    self.debug_point(context, max_loc, type="CUBE")
#                    self.debug_point(context, max_loc)
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            #dispersion *= 5
            tip_gather_ratio = dispersion * self.tip_gather_ratio * 0.01
            create_joint(centers, tips, bounds, rs, ms, tip_gather_ratio, name, chest, bone_type, end=end_loc)
        else:
            tips = []
            for joint in joints:
                stem = end_loc - neck_loc
                branch = joint - neck_loc
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                limit_angle = self.any_limit_angle
                close = stem.dot(branch)
                if 0 < close and limit_angle < angle:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    for center in centers:
                        x, y, z = center
                        current = Vector((self.lerp(ri, le, x * mx), self.lerp(fr, ba, y * my), self.lerp(bo, to, z * mz)))
                        close =  (current - joint).dot(branch_normal)
                        if max_close < close:
                            max_close = close
                            max_loc = current
                    tips.append((joint, max_loc))
#                    self.debug_point(context, max_loc, type="CUBE")
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            #left_hands = create_joint(centers, tips, bounds, rs, ms, gather_ratio, name, center_bone, BONE_TYPE_ANY)
            if (bone_type == BONE_TYPE_ARM_LEFT or bone_type == BONE_TYPE_ARM_RIGHT) and context.window_manager.love2d3d.armature_finger:
                pass
            else:
                create_joint(centers, tips, bounds, rs, ms, gather_ratio, name, end_bone, BONE_TYPE_ANY)
            if bone_type == BONE_TYPE_ARM_LEFT:
                left_hands.append(end_bone)
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                right_hands.append(end_bone)
        """
            Finger process.
        """
        if not finger and context.window_manager.love2d3d.armature_finger:
            #if left_hand is not None:
            #    fingers = []
            #    for polygon in mesh.polygons:
            #        center =  mat * Vector(polygon.center)
            #        vec = left_hand.tail - left_hand.head
            #        coeff = (center - left_hand.head.lerp(left_hand.tail, 0.5)).dot(vec)
            #        #coeff = (center - hand_center).dot(vec)                
            #        if 0 < coeff:
            #            fingers.append(polygon)
            #    if len(fingers) != 0:
            #        bone = armature.data.edit_bones.new("head")
            #        self.create_bone(context, armature, bone, obj, left_hand, bones, bone_type=BONE_TYPE_FINGER, fingers=fingers)
            self.crete_finger(context, armature, obj, mesh, mat, bones, left_hands, BONE_TYPE_FINGER_LEFT)
            self.crete_finger(context, armature, obj, mesh, mat, bones, right_hands, BONE_TYPE_FINGER_RIGHT)
            #self.crete_finger(context, armature, obj, mesh, mat, bones, left_foots)
            #self.crete_finger(context, armature, obj, mesh, mat, bones, right_foots)

        """
            Finish process.
        """ 
        if not finger:
            bpy.data.meshes.remove(mesh)
        return start_bone, end_bone
    def index_bone(self, location):
        y = int(location.y * 1000)
        return str(y)

    def crete_finger(self, context, armature, obj, mesh, mat, bones, hands, bone_type):
        for hand in hands:
            fingers = []
            for polygon in mesh.polygons:
                center =  mat * Vector(polygon.center)
                vec = hand.tail - hand.head
                m = hand.head.lerp(hand.tail, 0.0)
                coeff = (center - m).dot(vec)
                #coeff = (center - hand_center).dot(vec)
                vec1 = center - hand.head
                if vec.length_squared == 0.0 or vec1.length_squared == 0.0:
                    continue
                angle = vec.angle(vec1)
                if 0 < coeff and angle < self.hand_limit_angle:
                    fingers.append(polygon)
            if len(fingers) != 0:
                bone = armature.data.edit_bones.new("head")
                self.create_bone(context, armature, bone, obj, hand, bones, bone_type=bone_type, fingers=fingers)

    def gather_point(self, points, bound, m, dispersion, parent=None, end=Vector((0, 0, 0))):
        """
            Gathering points to gathers.
        """
        gathers = []
        alreadys = [False for p in points]
        for k, point in enumerate(points):            
            if alreadys[k]:
                continue
            hits = [k,]
            self._gather_point(k, points, bound, m, dispersion, hits, parent=parent, end=end)
            gathers.append(hits)
            for hit in hits:
                alreadys[hit] = True
        return gathers

    def _gather_point(self, index, points, bound, m, dispersion, hits, parent=None, end=Vector((0, 0, 0))):
        """
            Recursion call of points' collision.
        """
        point = points[index]
        current_count = len(hits)
        px, py, pz = point
        le, ri, ba, fr, to, bo = bound
        mx, my, mz = m
        if parent is not None:
            vec = (end - parent.head).normalized()
            p_loc = Vector((self.lerp(ri, le, px * mx), self.lerp(fr, ba, py * my), self.lerp(bo, to, pz * mz)))
            coeff = (p_loc- parent.head).dot(vec)
            p_loc = (p_loc- parent.head) - coeff * vec
        else:
            p_loc = Vector((self.lerp(ri, le, px * mx), self.lerp(fr, ba, py * my), self.lerp(bo, to, pz * mz)))
        neighbors = []
        for k, neighbor in enumerate(points):
            if neighbor == point:
                continue
            nx, ny, nz = neighbor
            if parent is not None:
                vec = (end - parent.head).normalized()
                n_loc = Vector((self.lerp(ri, le, nx * mx), self.lerp(fr, ba, ny * my), self.lerp(bo, to, nz * mz)))
                coeff = (n_loc - parent.head).dot(vec)
                n_loc = (n_loc - parent.head) - coeff * vec
            else:
                n_loc = Vector((self.lerp(ri, le, nx * mx), self.lerp(fr, ba, ny * my), self.lerp(bo, to, nz * mz)))
            length = (p_loc - n_loc).length_squared
            #length = (p_loc - n_loc).length_squared
            if length < dispersion:
                neighbors.append(k)
        for neighbor in neighbors:
            already = False
            for hit in hits:
                already = already or hit == neighbor
            if not already:
                hits.append(neighbor)
        if current_count == len(hits):
            return True
        for neighbor in neighbors:
            g = self._gather_point(neighbor, points, bound, m, dispersion, hits, parent=parent, end=end)
            if g:
                return True

    def create_head(self, bone, obj, chest):
        b = self.bound_loc(obj)
        p = b[BOUND_TOP]
        n = b[BOUND_BOTTOM]
        bone.head = Vector((obj.location.x, obj.location.y, n))
        bone.tail = Vector((obj.location.x, obj.location.y, p))
        bone.parent = chest
        return bone

        
        
class VIEW3D_PT_love2d3d(bpy.types.Panel):

    bl_label = "Love2D3D"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Create"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Object", icon="OBJECT_DATA")
        col.operator(CreateObject.bl_idname, text="Create")
        
        #row = col.row()
        col = layout.column(align=True)
        col.label(text="Preview",icon="GHOST_ENABLED")
        preview = context.window_manager.love2d3d.preview
        col.operator(Preview.bl_idname, text="On" if preview else "Off")
        col = layout.column(align=True)
        col.label(text="Image", icon="IMAGE_DATA")
        col.operator("image.open", icon="FILESEL", text="Open")
        col = layout.column(align=True)
        col.prop_search(context.window_manager.love2d3d,
                        "image_front", context.blend_data, "images")
        col.prop_search(context.window_manager.love2d3d,
                        "image_back", context.blend_data, "images")
        #layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Separation", icon="IMAGE_RGB_ALPHA")
        #col.prop(context.window_manager.love2d3d, "threshold")
        #col.prop(context.window_manager.love2d3d, "opacity")
        #layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Geometry", icon="EDITMODE_HLT")
        #col.prop(context.window_manager.love2d3d, "view_align")
        #col.prop(context.window_manager.love2d3d, "depth_front")
        #col.prop(context.window_manager.love2d3d, "depth_back")
        #col.prop(context.window_manager.love2d3d, "scale")
        #layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Quality", icon="MOD_SMOOTH")
        #col.prop(context.window_manager.love2d3d, "rough")
        #col.prop(context.window_manager.love2d3d, "smooth")
        #col.prop(context.window_manager.love2d3d, "fat")
        #layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Decimate", icon="MOD_DECIM")        
        #col.prop(context.window_manager.love2d3d, "decimate")
        #col.prop(context.window_manager.love2d3d, "decimate_ratio")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Armature", icon="ARMATURE_DATA")        
        col.operator(CreateArmature.bl_idname, text="Create")
        col = layout.column(align=True)
        col.prop(context.window_manager.love2d3d, "armature_finger")
        col.prop(context.window_manager.love2d3d, "armature_resolution")
        col.prop(context.window_manager.love2d3d, "armature_finger_resolution")
        #layout.separator()
        #col = layout.column(align=True)
        #col.label(text="Option", icon="SCRIPTWIN")
        #col.prop(context.window_manager.love2d3d, "modifier")
        #col.prop(context.window_manager.love2d3d, "shadeless")


class Love2D3DProps(bpy.types.PropertyGroup):
    image_front = bpy.props.StringProperty(name="Front",
                                           description="Front image of mesh")
    image_back = bpy.props.StringProperty(name="Back",
                                          description="Back image of mesh")
    rough = bpy.props.IntProperty(name="Rough",
                                  description="Roughness of image", min=1,
                                  default=8, subtype="PIXEL")
    smooth = bpy.props.IntProperty(name="Smooth",
                                   description="Smoothness of mesh",
                                   min=1, default=30)
    scale = bpy.props.FloatProperty(name="Scale",
                                    description="Length per pixel",
                                    unit="LENGTH", min=0.001, default=0.01, precision=4)
    #depth_front = bpy.props.FloatProperty(name="Front",
    #                                      description="Depth of front face",
    #                                      unit="LENGTH", min=0, default=40)
    #depth_back = bpy.props.FloatProperty(name="Back",
    #                                     description="Depth of back face",
    #                                     unit="LENGTH", min=0, default=40)
    depth_front = bpy.props.FloatProperty(name="Front",
                                          description="Depth of front face",
                                          unit="NONE", min=0, default=1)
    depth_back = bpy.props.FloatProperty(name="Back",
                                         description="Depth of back face",
                                         unit="NONE", min=0, default=1)
    fat = bpy.props.FloatProperty(name="Fat",
                                  description="Fat of mesh",
                                  default=0.2, min=0.0)
    modifier = bpy.props.BoolProperty(name="Modifier",
                                      description="Apply modifiers to object",
                                      default=True)
    threshold = bpy.props.FloatProperty(name="Threshold",
                                        description="Threshold of background in image",
                                        min=0.0, max=1.0,
                                        default=0.0, subtype="FACTOR")
    opacity = bpy.props.BoolProperty(name="Opacity",
                                     description="Use Opacity for threshold")
    view_align = bpy.props.BoolProperty(name="View align",
                                     description="Use view align for mesh")
    preview = bpy.props.BoolProperty(name="Preview",
                                     description="Use preview for mesh now",
                                     options={'HIDDEN'})
    decimate = bpy.props.BoolProperty(name="Decimate",
                                      description="Use decimate modifier to object",
                                      default=False)
    decimate_ratio = bpy.props.FloatProperty(name="Ratio",
                                  description="Decimate ratio",
                                  default=0.2, min=0.0, max=1.0, subtype="FACTOR")
    shadeless = bpy.props.BoolProperty(name="Shadeless",
                                      description="Use shadeless in object's material",
                                      default=True)
    armature_resolution = bpy.props.FloatProperty(name="Resolution",
                                          description="Resolution of armature calculation",
                                          min=1, default=6.0)
    armature_finger_resolution = bpy.props.FloatProperty(name="Finger resolution",
                                          description="Finger's resolution of armature calculation",
                                          min=1, default=6.0)
    armature_finger = bpy.props.BoolProperty(name="Finger",
                                      description="Use finger in armature",
                                      default=False)

def register():
    bpy.utils.register_module(__name__)
    bpy.types.WindowManager.love2d3d \
        = bpy.props.PointerProperty(type=Love2D3DProps)


def unregister():
    del bpy.types.WindowManager.love2d3d
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
