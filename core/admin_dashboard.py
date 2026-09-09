from core.account_manager import search_users, user_usage, add_credits, set_disabled, stats, get_user


def dashboard_stats():
    s = stats()
    return (f"### 管理员统计\n\n"
            f"**用户：** {s['users']}　 **正常：** {s['active']}　 **剩余总额度：** {s['credits']}\n\n"
            f"**总任务：** {s['total']}　 **成功：** {s['success']}　"
            f"**Gemini 消耗：** {s['gemini']}　 **OpenAI 消耗：** {s['openai']}　 **本地任务：** {s['local']}")


def require_admin(user):
    if not user or not user.get('is_admin'):
        raise PermissionError('需要管理员权限')


def find_users(keyword=''):
    return search_users(keyword)


def details(user_id):
    return user_usage(int(user_id))


def adjust(user_id, amount):
    return add_credits(int(user_id), int(amount), 'admin adjustment')


def toggle(user_id):
    u = get_user(int(user_id))
    if not u:
        raise ValueError('用户不存在')
    set_disabled(u['id'], not bool(u['disabled']))
    return get_user(u['id'])
