import time

from bin.rw.get_msg import get_msg


def read_posts(vkapp, group_list, count, offset=0):
    posts = []
    for group in group_list.values():
        try:
            posts += get_msg(vkapp, group, offset, count)
        except:
            print('Не получил посты - ', group)
        time.sleep(1)
    return posts


if __name__ == '__main__':
    pass
